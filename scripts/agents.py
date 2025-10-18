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


# ===============================
# STATE PROCESSING FUNCTIONS
# ===============================

def discretize_state(state):
    """
    Discretize continuous state into bins for tabular methods.
    
    State: [capacity, onboard, station_idx, direction, hour, minute]
    
    Features:
    - Cap bins to prevent infinite state space
    - Use minute information in time discretization
    - Reduce state space size for better convergence
    """
    cap, onboard, station_idx, direction, hour, minute = state
    
    # Cap capacity bins at reasonable max (20 bins = 2000 capacity)
    cap_bin = min(int(cap // 100), 20)
    
    # Cap onboard bins at reasonable max (10 bins = 500 passengers)
    on_bin = min(int(onboard // 50), 10)
    
    # Direction: 0 or 1
    dir_bin = 1 if direction >= 0 else 0
    
    # Time discretization: combine hour and minute into time periods
    # 18 operating hours (4:00-22:00) divided into 6 periods of 3 hours each
    time_minutes = hour * 60 + minute
    operating_start = 4 * 60  # 4:00 AM
    minutes_since_start = max(0, time_minutes - operating_start)
    time_period = min(int(minutes_since_start // 180), 5)  # 3-hour blocks, max 5
    
    return (cap_bin, on_bin, int(station_idx), dir_bin, time_period)


def normalize_state(state):
    """Normalize state for neural network agents"""
    cap, onboard, station_idx, direction, hour, minute = state
    return np.array([
        cap / 1000,              # Normalize capacity (assume max ~1000)
        onboard / 500,           # Normalize passengers (assume max ~500)
        station_idx / 12,        # Normalize station index (0-12)
        (direction + 1) / 2,     # Convert -1/1 to 0/1
        hour / 23,               # Normalize hour (0-23)
        minute / 59              # Normalize minute (0-59)
    ], dtype=np.float32)


# ===============================
# MONTE CARLO AGENT
# ===============================

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
        """Select action using epsilon-greedy policy"""
        ds = discretize_state(state)
        if (not greedy) and (random.random() < self.eps):
            return random.randint(0, self.n_actions-1)
        qvals = [self.Q[(ds, a)] for a in range(self.n_actions)]
        # If no Q-values, prefer no-action (2) initially
        if all(q == 0 for q in qvals):
            return 2
        return int(np.argmax(qvals))

    def update(self, episode):
        """First-visit Monte Carlo update"""
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
        
        # Decay epsilon
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ===============================
# Q-LEARNING AGENT
# ===============================

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
        """Select action using epsilon-greedy policy"""
        ds = discretize_state(state)
        if (not greedy) and (random.random() < self.eps):
            return random.randint(0, self.n_actions-1)
        qvals = [self.Q[(ds, a)] for a in range(self.n_actions)]
        # If no Q-values, prefer no-action (2) initially
        if all(q == 0 for q in qvals):
            return 2
        return int(np.argmax(qvals))

    def update(self, s, a, r, s_next, terminated, truncated):
        """Q-Learning TD update"""
        ds = discretize_state(s)
        ds_next = discretize_state(s_next)
        
        done = terminated or truncated
        
        if done:
            target = r
        else:
            best_next = max([self.Q[(ds_next, a2)] for a2 in range(self.n_actions)])
            target = r + self.gamma * best_next
            
        self.Q[(ds, a)] += self.alpha * (target - self.Q[(ds, a)])
        
        # Decay learning rate and epsilon
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ===============================
# ACTOR-CRITIC AGENT
# ===============================

class ACNetwork(nn.Module):
    """Actor-Critic network with shared feature extraction."""
    
    def __init__(self, state_dim=6, action_dim=3, hidden=128):
        super().__init__()
        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        
        # Actor head
        self.actor = nn.Linear(hidden, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


class ActorCriticAgent:
    """Actor-Critic agent with advantage estimation."""
    
    def __init__(self, state_dim=6, action_dim=3, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.net = ACNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.state_dim = state_dim
        self.action_dim = action_dim

    def policy(self, state, greedy=False):
        """Get action from policy."""
        normalized_state = normalize_state(state)
        st = torch.FloatTensor(normalized_state).unsqueeze(0)
        
        with torch.no_grad() if greedy else torch.enable_grad():
            probs, val = self.net(st)
        
        if greedy:
            a = torch.argmax(probs, dim=-1).item()
            return a, None, None
        
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a), val

    def learn(self, trajectory):
        """Learn from a trajectory using advantage actor-critic."""
        if len(trajectory) == 0:
            return
        
        returns = []
        G = 0
        
        # Calculate discounted returns
        for _, _, r, _, _, _ in reversed(trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Extract components and recalculate forward pass
        states = []
        actions = []
        
        for state, action_info, _, _, _, _ in trajectory:
            states.append(normalize_state(state))
            if len(action_info) == 3:
                action, _, _ = action_info
                actions.append(action)
        
        if not states:
            return
        
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        
        # Forward pass
        probs, values = self.net(states_tensor)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        
        values = values.squeeze()
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Calculate losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = -self.entropy_coef * entropy
        
        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Return losses for logging
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }
