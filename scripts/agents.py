"""
RL Agent Implementations for Train Game

This module contains implementations of three RL agents:
- MonteCarloAgent: First-visit Monte Carlo with epsilon-greedy
- QLearningAgent: Q-Learning with learning rate and epsilon decay
- ActorCriticAgent: Deep Actor-Critic with PyTorch
"""

import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def discretize_state(state):
    """
    Discretize continuous state into bins for tabular methods.
    """
    cap, onboard, station_idx, direction, hour, minute = state
    cap_bin = min(int(cap // 100), 20)
    on_bin = min(int(onboard // 50), 10)
    dir_bin = 1 if direction >= 0 else 0
    time_minutes = hour * 60 + minute
    operating_start = 4 * 60
    minutes_since_start = max(0, time_minutes - operating_start)
    time_period = min(int(minutes_since_start // 180), 5)
    return (cap_bin, on_bin, int(station_idx), dir_bin, time_period)


def normalize_state(state):
    """Normalize state for neural network agents with clipping for robustness."""
    cap, onboard, station_idx, direction, hour, minute = state
    
    # Clip values to the expected range before normalizing
    norm_cap = np.clip(cap / 1000, 0, 1)
    norm_onboard = np.clip(onboard / 500, 0, 1)
    
    return np.array([
        norm_cap,
        norm_onboard,
        station_idx / 12,
        (direction + 1) / 2,
        hour / 23,
        minute / 59
    ], dtype=np.float32)

class MonteCarloAgent:
    """First-visit Monte Carlo agent with epsilon-greedy policy."""
    def __init__(self, n_actions=3, eps=0.1, gamma=0.99, eps_decay=0.995, eps_min=0.01):
        self.n_actions = n_actions
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.Q = defaultdict(float)
        self.returns = defaultdict(list)

    def policy(self, state, greedy=False):
        ds = discretize_state(state)
        if (not greedy) and (random.random() < self.eps):
            return random.randint(0, self.n_actions - 1)
        qvals = [self.Q[(ds, a)] for a in range(self.n_actions)]
        if all(q == 0 for q in qvals):
            return 2
        return int(np.argmax(qvals))

    def update(self, episode):
        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = r + self.gamma * G
            ds = discretize_state(s)
            key = (ds, a)
            if key not in visited:
                visited.add(key)
                self.returns[key].append(G)
                self.Q[key] = np.mean(self.returns[key])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

class QLearningAgent:
    """Q-Learning agent with epsilon-greedy policy and learning rate decay."""
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.99, eps=0.1, 
                 eps_decay=0.995, eps_min=0.01, alpha_decay=0.9999, alpha_min=0.01):
        self.n_actions = n_actions
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.Q = defaultdict(float)

    def policy(self, state, greedy=False):
        ds = discretize_state(state)
        if (not greedy) and (random.random() < self.eps):
            return random.randint(0, self.n_actions - 1)
        qvals = [self.Q[(ds, a)] for a in range(self.n_actions)]
        if all(q == 0 for q in qvals):
            return 2
        return int(np.argmax(qvals))

    def update(self, s, a, r, s_next, terminated, truncated):
        ds = discretize_state(s)
        ds_next = discretize_state(s_next)
        done = terminated or truncated
        if done:
            target = r
        else:
            best_next = max([self.Q[(ds_next, a2)] for a2 in range(self.n_actions)])
            target = r + self.gamma * best_next
        self.Q[(ds, a)] += self.alpha * (target - self.Q[(ds, a)])
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

# In agents.py, replace the ACNetwork class with this one

class ACNetwork(nn.Module):
    """A deeper and wider Actor-Critic network for more complex tasks."""
    def __init__(self, state_dim=6, action_dim=3, hidden=256): # Increased hidden size
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), # Added another layer
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2), # Added another layer
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Linear(hidden // 2, action_dim)
        # Critic head
        self.critic = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        x = self.shared_layer(x)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class ActorCriticAgent:
    """Actor-Critic agent adapted for online, step-by-step learning."""
    def __init__(self, state_dim=6, action_dim=3, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.net = ACNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def policy(self, state):
        """Selects an action, returning the action, its log probability, and the state value."""
        normalized_state = normalize_state(state)
        st = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.net(st)
        
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value

    def learn(self, state, action_data, reward, next_state, terminated, truncated):
        """Performs a learning update from a single step of experience."""
        self.optimizer.zero_grad()
        
        _action, log_prob, value = action_data
        done = terminated or truncated

        # Prepare tensors for calculation
        next_normalized_state = normalize_state(next_state)
        next_st = torch.FloatTensor(next_normalized_state).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([reward]).to(self.device)

        # Get the value of the next state
        with torch.no_grad():
            _logits, next_value = self.net(next_st)
        
        # If the episode is over, the value of the "next state" is 0
        if done:
            next_value = torch.FloatTensor([0.0]).to(self.device)

        # --- FINAL FIX IS HERE ---
        # Use .squeeze(-1) to ensure the tensors become shape [1] instead of a scalar
        value = value.squeeze(-1)
        next_value = next_value.squeeze(-1)

        # Calculate the TD target and advantage
        td_target = reward_t + self.gamma * next_value
        advantage = td_target - value
        
        # --- Calculate Losses ---
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = nn.MSELoss()(value, td_target)

        # Re-calculate distribution for entropy
        normalized_state = normalize_state(state)
        st = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
        logits, _ = self.net(st)
        entropy = Categorical(logits=logits).entropy()
        
        entropy_loss = -self.entropy_coef * entropy
        
        # Combine losses and update the network
        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()