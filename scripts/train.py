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
import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import pickle

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
    """Train Monte Carlo agent and return detailed history."""
    agent = MonteCarloAgent(
        n_actions=config.N_ACTIONS,
        **config.MONTE_CARLO_CONFIG
    )
    
    history = []
    
    for ep in range(episodes):
        state, info = env.reset()
        episode_steps = []
        ep_reward = 0
        
        while True:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Record step data
            episode_steps.append({
                'state': state.tolist(),
                'action': action,
                'reward': reward,
                'next_state': next_state.tolist(),
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # For MC, the update uses the entire recorded episode (correct for this algo)
        mc_episode_format = [(s['state'], s['action'], s['reward']) for s in episode_steps]
        agent.update(mc_episode_format)
        
        final_score, _ = env.final_score()
        
        history.append({
            'episode': ep,
            'score': final_score,
            'config_cost': info['total_config_cost'],
            'total_reward': ep_reward,
            'steps': episode_steps
        })
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = [h['score'] for h in history[-100:]]
            print(f"[MC] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f} | "
                  f"Epsilon: {agent.eps:.3f}")
    
    return agent, history


def train_qlearning(env, episodes, verbose=True):
    """Train Q-Learning agent and return detailed history."""
    agent = QLearningAgent(
        n_actions=config.N_ACTIONS,
        **config.Q_LEARNING_CONFIG
    )
    
    history = []
    
    for ep in range(episodes):
        state, info = env.reset()
        episode_steps = []
        ep_reward = 0
        
        while True:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning updates after every step (online)
            agent.update(state, action, reward, next_state, terminated, truncated)
            
            episode_steps.append({
                'state': state.tolist(),
                'action': action,
                'reward': reward,
                'next_state': next_state.tolist(),
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        final_score, _ = env.final_score()
        
        history.append({
            'episode': ep,
            'score': final_score,
            'config_cost': info['total_config_cost'],
            'total_reward': ep_reward,
            'steps': episode_steps
        })
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = [h['score'] for h in history[-100:]]
            print(f"[QL] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f} | "
                  f"Epsilon: {agent.eps:.3f} | Alpha: {agent.alpha:.4f}")
    
    return agent, history


def train_actor_critic(env, episodes, verbose=True):
    """Train Actor-Critic agent and return detailed history."""
    agent = ActorCriticAgent(**config.ACTOR_CRITIC_CONFIG)
    
    history = []
    
    for ep in range(episodes):
        state, info = env.reset()
        episode_steps = []
        ep_reward = 0
        
        while True:
            action, log_prob, value = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # --- CRITICAL FIX ---
            # Learn from the experience at each step (online learning).
            # This provides immediate feedback and leads to more stable training.
            # NOTE: Your ActorCriticAgent.learn() method must be modified to accept
            # a single transition instead of a full trajectory.
            agent.learn(state, (action, log_prob, value), reward, next_state, terminated, truncated)
            
            # For logging
            episode_steps.append({
                'state': state.tolist(),
                'action': action,
                'reward': reward,
                'next_state': next_state.tolist(),
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        final_score, _ = env.final_score()
        
        history.append({
            'episode': ep,
            'score': final_score,
            'config_cost': info['total_config_cost'],
            'total_reward': ep_reward,
            'steps': episode_steps
        })
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = [h['score'] for h in history[-100:]]
            print(f"[AC] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f}")
    
    return agent, history


def save_agent(agent, agent_name, save_dir):
    """Save trained agent to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(agent, ActorCriticAgent):
        filepath = save_dir / f"{agent_name}_model.pt"
        torch.save(agent.net.state_dict(), filepath)
    else:
        # For tabular agents, we save the agent object itself
        filepath = save_dir / f"{agent_name}_agent.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(agent, f)
    
    print(f"Saved {agent_name} to {filepath}")


def save_training_logs(results, save_dir):
    """Save detailed training logs to a JSON file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = save_dir / 'training_logs.json'
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(log_file, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
    
    print(f"Saved detailed training logs to {log_file}")


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
                       help='Directory to save trained agents and logs')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save trained agents or logs')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress training progress output')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    env = TrainGameEnv()
    verbose = not args.quiet
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Agent(s): {args.agent}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    print(f"  Save Dir: {args.save_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    if args.agent in ['montecarlo', 'all']:
        print("\n🎲 Training Monte Carlo Agent...")
        mc_agent, mc_history = train_monte_carlo(env, args.episodes, verbose)
        all_results['montecarlo'] = mc_history
        if not args.no_save:
            save_agent(mc_agent, 'monte_carlo', args.save_dir)
    
    if args.agent in ['qlearning', 'all']:
        print("\n🧠 Training Q-Learning Agent...")
        q_agent, q_history = train_qlearning(env, args.episodes, verbose)
        all_results['qlearning'] = q_history
        if not args.no_save:
            save_agent(q_agent, 'q_learning', args.save_dir)
    
    if args.agent in ['actorcritic', 'all']:
        print("\n🎭 Training Actor-Critic Agent...")
        ac_agent, ac_history = train_actor_critic(env, args.episodes, verbose)
        all_results['actorcritic'] = ac_history
        if not args.no_save:
            save_agent(ac_agent, 'actor_critic', args.save_dir)
    
    elapsed_time = time.time() - start_time
    
    # --- ADD THIS DEBUGGING BLOCK ---
    if 'actorcritic' in all_results:
        print("\n" + "="*20 + " DEBUGGING ACTOR-CRITIC DATA " + "="*20)
        ac_history = all_results['actorcritic']
        last_episode = ac_history[-1]
        
        # Check 1: Do we have the correct number of episodes?
        print(f"Number of logged episodes for AC: {len(ac_history)}")
        
        # Check 2: What was the final score in the data?
        print(f"Final logged score for AC: {last_episode['score']}")
        
        # Check 3: What do the action counts for the LAST episode look like?
        if last_episode['steps']:
            action_counts = pd.Series([s['action'] for s in last_episode['steps']]).value_counts()
            print("Action counts for the final AC episode:")
            print(action_counts)
        print("="*66 + "\n")
    # --- END OF DEBUGGING BLOCK ---
    
    if not args.no_save and all_results:
        save_training_logs(all_results, args.save_dir)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! ({elapsed_time:.1f}s)")
    print(f"{'='*60}\n")
    
    for agent_name, history in all_results.items():
        scores = [h['score'] for h in history]
        costs = [h['config_cost'] for h in history]
        
        final_scores = scores[-100:]
        final_costs = costs[-100:]
        
        print(f"{agent_name.upper()}:")
        print(f"  Final Score (last 100): {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}")
        print(f"  Config Cost (last 100): {np.mean(final_costs):.1f}")
        print()

if __name__ == '__main__':
    main()