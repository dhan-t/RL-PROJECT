"""
Training script for RL agents on Train Game environment.

Usage:
    python train.py --agent qlearning --episodes 1000 --seed 42
    python train.py --agent all --episodes 500
    python train.py --agent actorcritic --episodes 2000 --save-dir ../my_models
"""

import sys
import os
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path to import train_game_env
sys.path.append(str(Path(__file__).parent.parent))
from train_game_env import TrainGameEnv

from agents import MonteCarloAgent, QLearningAgent, ActorCriticAgent
import config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_monte_carlo(env, episodes, verbose=True):
    """Train Monte Carlo agent."""
    agent = MonteCarloAgent(
        n_actions=config.N_ACTIONS,
        **config.MONTE_CARLO_CONFIG
    )
    
    scores = []
    config_costs = []
    total_rewards = []
    
    for ep in range(episodes):
        state, info = env.reset()
        episode = []
        ep_reward = 0
        
        while True:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode.append((state, action, reward))
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        agent.update(episode)
        
        # Calculate final score
        final_score, _ = env.final_score()
        scores.append(final_score)
        config_costs.append(info['total_config_cost'])
        total_rewards.append(ep_reward)
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = scores[-100:]
            print(f"[MC] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f} | "
                  f"Epsilon: {agent.eps:.3f}")
    
    return agent, {
        'scores': scores,
        'config_costs': config_costs,
        'total_rewards': total_rewards
    }


def train_qlearning(env, episodes, verbose=True):
    """Train Q-Learning agent."""
    agent = QLearningAgent(
        n_actions=config.N_ACTIONS,
        **config.Q_LEARNING_CONFIG
    )
    
    scores = []
    config_costs = []
    total_rewards = []
    
    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0
        
        while True:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.update(state, action, reward, next_state, terminated, truncated)
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Calculate final score
        final_score, _ = env.final_score()
        scores.append(final_score)
        config_costs.append(info['total_config_cost'])
        total_rewards.append(ep_reward)
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = scores[-100:]
            print(f"[QL] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f} | "
                  f"Epsilon: {agent.eps:.3f} | Alpha: {agent.alpha:.4f}")
    
    return agent, {
        'scores': scores,
        'config_costs': config_costs,
        'total_rewards': total_rewards
    }


def train_actor_critic(env, episodes, verbose=True):
    """Train Actor-Critic agent."""
    agent = ActorCriticAgent(**config.ACTOR_CRITIC_CONFIG)
    
    scores = []
    config_costs = []
    total_rewards = []
    
    for ep in range(episodes):
        state, info = env.reset()
        trajectory = []
        ep_reward = 0
        
        while True:
            action, log_prob, value = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append((state, (action, log_prob, value), reward, 
                             next_state, terminated, truncated))
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        agent.learn(trajectory)
        
        # Calculate final score
        final_score, _ = env.final_score()
        scores.append(final_score)
        config_costs.append(info['total_config_cost'])
        total_rewards.append(ep_reward)
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = scores[-100:]
            print(f"[AC] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f}")
    
    return agent, {
        'scores': scores,
        'config_costs': config_costs,
        'total_rewards': total_rewards
    }


def save_agent(agent, agent_name, save_dir):
    """Save trained agent to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(agent, ActorCriticAgent):
        # Save PyTorch neural network weights
        filepath = save_dir / f"{agent_name}_model.pt"
        torch.save(agent.net.state_dict(), filepath)
    else:
        # Save tabular agents (Q-tables) as pickle
        # Note: .pkl format required for dictionary-based agents
        filepath = save_dir / f"{agent_name}_model.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(agent, f)
    
    print(f"Saved {agent_name} to {filepath}")


def save_training_history(results, save_dir):
    """Save training history for visualization."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history_file = save_dir / 'training_history.pkl'
    with open(history_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved training history to {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train RL agents on Train Game')
    parser.add_argument('--agent', type=str, default='qlearning',
                       choices=['montecarlo', 'qlearning', 'actorcritic', 'all'],
                       help='Which agent to train (default: qlearning)')
    parser.add_argument('--episodes', type=int, default=config.DEFAULT_EPISODES,
                       help=f'Number of training episodes (default: {config.DEFAULT_EPISODES})')
    parser.add_argument('--seed', type=int, default=config.DEFAULT_SEED,
                       help=f'Random seed (default: {config.DEFAULT_SEED})')
    parser.add_argument('--save-dir', type=str, default=config.SAVED_AGENTS_DIR,
                       help='Directory to save trained agents')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save trained agents')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress training progress output')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seed(args.seed)
    
    # Create environment
    env = TrainGameEnv()
    
    verbose = not args.quiet
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Agent(s): {args.agent}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    print(f"  Save Dir: {args.save_dir}")
    print(f"{'='*60}\n")
    
    # Training
    start_time = time.time()
    
    if args.agent in ['montecarlo', 'all']:
        print("\n🎲 Training Monte Carlo Agent...")
        mc_agent, mc_results = train_monte_carlo(env, args.episodes, verbose)
        results['montecarlo'] = mc_results
        if not args.no_save:
            save_agent(mc_agent, 'monte_carlo', args.save_dir)
    
    if args.agent in ['qlearning', 'all']:
        print("\n🧠 Training Q-Learning Agent...")
        q_agent, q_results = train_qlearning(env, args.episodes, verbose)
        results['qlearning'] = q_results
        if not args.no_save:
            save_agent(q_agent, 'q_learning', args.save_dir)
    
    if args.agent in ['actorcritic', 'all']:
        print("\n🎭 Training Actor-Critic Agent...")
        ac_agent, ac_results = train_actor_critic(env, args.episodes, verbose)
        results['actorcritic'] = ac_results
        if not args.no_save:
            save_agent(ac_agent, 'actor_critic_best', args.save_dir)
    
    elapsed_time = time.time() - start_time
    
    # Save training history for visualization
    if not args.no_save and results:
        save_training_history(results, args.save_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Complete! ({elapsed_time:.1f}s)")
    print(f"{'='*60}\n")
    
    for agent_name, agent_results in results.items():
        scores = agent_results['scores']
        costs = agent_results['config_costs']
        rewards = agent_results['total_rewards']
        
        # Take last 100 episodes for final performance
        final_scores = scores[-100:]
        final_costs = costs[-100:]
        final_rewards = rewards[-100:]
        
        print(f"{agent_name.upper()}:")
        print(f"  Final Score: {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}")
        print(f"  Config Cost: {np.mean(final_costs):.1f}")
        print(f"  Total Reward: {np.sum(final_rewards):,.0f}")
        print()


if __name__ == '__main__':
    main()
