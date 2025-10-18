"""
Interactive demo script - play the train game with a trained agent.

Usage:
    python play_demo.py --agent qlearning
    python play_demo.py --agent-path ../saved_agents/q_learning_agent.pkl --episodes 5
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from train_game_env import TrainGameEnv

from agents import MonteCarloAgent, QLearningAgent, ActorCriticAgent
import config


def load_agent(filepath):
    """Load a trained agent from file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Agent file not found: {filepath}")
    
    if filepath.suffix == '.pt':
        # Load Actor-Critic model
        agent = ActorCriticAgent(**config.ACTOR_CRITIC_CONFIG)
        agent.net.load_state_dict(torch.load(filepath))
        agent.net.eval()
        return agent, 'actorcritic'
    elif filepath.suffix == '.pkl':
        # Load tabular agent
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        
        if isinstance(agent, MonteCarloAgent):
            return agent, 'montecarlo'
        elif isinstance(agent, QLearningAgent):
            return agent, 'qlearning'
        else:
            return agent, 'unknown'
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")


def play_episode(env, agent, verbose=True):
    """Play one episode and display step-by-step actions."""
    state, info = env.reset()
    ep_reward = 0
    step = 0
    
    print("\n" + "="*80)
    print("STARTING NEW EPISODE")
    print("="*80)
    print(f"Initial State:")
    print(f"  Capacity: {state[0]:.0f}")
    print(f"  Onboard: {state[1]:.0f}")
    print(f"  Station: {state[2]:.0f}")
    print(f"  Direction: {'Northbound' if state[3] >= 0 else 'Southbound'}")
    print(f"  Time: {int(state[4]):02d}:{int(state[5]):02d}")
    print()
    
    while True:
        # Get action
        if isinstance(agent, ActorCriticAgent):
            action, _, _ = agent.policy(state, greedy=True)
        else:
            action = agent.policy(state, greedy=True)
        
        action_name = config.ACTION_NAMES[action]
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        step += 1
        ep_reward += reward
        
        if verbose:
            print(f"Step {step}:")
            print(f"  Action: {action_name}")
            print(f"  Reward: {reward:.2f}")
            print(f"  New Capacity: {next_state[0]:.0f}")
            print(f"  Onboard: {next_state[1]:.0f}")
            print(f"  Station: {next_state[2]:.0f}")
            print(f"  Time: {int(next_state[4]):02d}:{int(next_state[5]):02d}")
            print()
        
        state = next_state
        
        if terminated or truncated:
            break
    
    # Calculate final score
    final_score, _ = env.final_score()
    
    print("="*80)
    print("EPISODE COMPLETE")
    print("="*80)
    print(f"Final Score: {final_score:.1f}")
    print(f"Total Config Cost: {info['total_config_cost']:.0f}")
    print(f"Episode Reward: {ep_reward:.2f}")
    print(f"Total Steps: {step}")
    print("="*80)
    print()
    
    return {
        'score': final_score,
        'config_cost': info['total_config_cost'],
        'reward': ep_reward,
        'steps': step
    }


def main():
    parser = argparse.ArgumentParser(description='Play Train Game with trained agent')
    parser.add_argument('--agent', type=str, default='qlearning',
                       choices=['montecarlo', 'qlearning', 'actorcritic'],
                       help='Which agent to use (default: qlearning)')
    parser.add_argument('--agent-path', type=str, default=None,
                       help='Path to specific agent file')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to play (default: 1)')
    parser.add_argument('--save-dir', type=str, default=config.SAVED_AGENTS_DIR,
                       help='Directory containing saved agents')
    parser.add_argument('--quiet', action='store_true',
                       help='Only show final results')
    
    args = parser.parse_args()
    
    # Load agent
    if args.agent_path:
        agent, agent_type = load_agent(args.agent_path)
        agent_name = Path(args.agent_path).stem
    else:
        save_dir = Path(args.save_dir)
        
        if args.agent == 'montecarlo':
            # Try new naming first, fall back to old naming
            filepath = save_dir / 'monte_carlo_model.pkl'
            if not filepath.exists():
                filepath = save_dir / 'monte_carlo_agent.pkl'
            agent_name = 'Monte Carlo'
        elif args.agent == 'qlearning':
            # Try new naming first, fall back to old naming
            filepath = save_dir / 'q_learning_model.pkl'
            if not filepath.exists():
                filepath = save_dir / 'q_learning_agent.pkl'
            agent_name = 'Q-Learning'
        elif args.agent == 'actorcritic':
            filepath = save_dir / 'actor_critic_best_model.pt'
            agent_name = 'Actor-Critic'
        
        agent, agent_type = load_agent(filepath)
    
    # Create environment
    env = TrainGameEnv()
    
    print(f"\n{'='*80}")
    print(f"TRAIN GAME DEMO - {agent_name.upper()}")
    print(f"{'='*80}")
    print(f"Playing {args.episodes} episode(s)")
    print()
    
    # Play episodes
    all_results = []
    
    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n>>> Episode {ep + 1}/{args.episodes}")
        
        results = play_episode(env, agent, verbose=not args.quiet)
        all_results.append(results)
    
    # Summary if multiple episodes
    if args.episodes > 1:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        scores = [r['score'] for r in all_results]
        costs = [r['config_cost'] for r in all_results]
        rewards = [r['reward'] for r in all_results]
        
        print(f"Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"Average Config Cost: {np.mean(costs):.1f}")
        print(f"Average Episode Reward: {np.mean(rewards):.1f}")
        print(f"Total Reward: {np.sum(rewards):.1f}")
        print("="*80)
        print()


if __name__ == '__main__':
    main()
