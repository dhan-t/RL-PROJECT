"""
Evaluation script for trained RL agents.

Usage:
    python evaluate.py --agent qlearning --episodes 100
    python evaluate.py --agent-path ../saved_agents/q_learning_agent.pkl
    python evaluate.py --all  # Evaluate all saved agents
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


def evaluate_agent(env, agent, episodes=100, verbose=False):
    """Evaluate an agent over multiple episodes."""
    scores = []
    config_costs = []
    total_rewards = []
    action_counts = {i: 0 for i in range(config.N_ACTIONS)}
    
    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0
        ep_actions = []
        
        while True:
            # Get greedy action
            if isinstance(agent, ActorCriticAgent):
                action, _, _ = agent.policy(state, greedy=True)
            else:
                action = agent.policy(state, greedy=True)
            
            ep_actions.append(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Calculate final score
        final_score, _ = env.final_score()
        scores.append(final_score)
        config_costs.append(info['total_config_cost'])
        total_rewards.append(ep_reward)
        
        # Count actions
        for a in ep_actions:
            action_counts[a] += 1
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Score: {scores[-1]:.1f} | Reward: {ep_reward:.1f}")
    
    # Calculate statistics
    results = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_config_cost': np.mean(config_costs),
        'total_reward': np.sum(total_rewards),
        'action_distribution': action_counts,
        'scores': scores,
        'config_costs': config_costs,
        'total_rewards': total_rewards
    }
    
    return results


def print_results(agent_name, results):
    """Print evaluation results in a nice format."""
    print(f"\n{'='*60}")
    print(f"Results for {agent_name.upper()}")
    print(f"{'='*60}")
    print(f"  Mean Score: {results['mean_score']:.1f} ± {results['std_score']:.1f}")
    print(f"  Mean Config Cost: {results['mean_config_cost']:.1f}")
    print(f"  Total Reward: {results['total_reward']:,.0f}")
    print(f"\n  Action Distribution:")
    
    total_actions = sum(results['action_distribution'].values())
    for action_idx, count in results['action_distribution'].items():
        action_name = config.ACTION_NAMES[action_idx]
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"    {action_name}: {count} ({pct:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument('--agent', type=str, default=None,
                       choices=['montecarlo', 'qlearning', 'actorcritic'],
                       help='Which agent to evaluate')
    parser.add_argument('--agent-path', type=str, default=None,
                       help='Path to specific agent file')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all saved agents')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--save-dir', type=str, default=config.SAVED_AGENTS_DIR,
                       help='Directory containing saved agents')
    parser.add_argument('--verbose', action='store_true',
                       help='Print episode-by-episode results')
    
    args = parser.parse_args()
    
    # Create environment
    env = TrainGameEnv()
    
    # Determine which agents to evaluate
    agents_to_eval = []
    
    if args.agent_path:
        # Load specific agent file
        agent, agent_type = load_agent(args.agent_path)
        agents_to_eval.append((Path(args.agent_path).stem, agent))
    elif args.all:
        # Load all agents in save directory
        save_dir = Path(args.save_dir)
        for filepath in save_dir.glob('*'):
            if filepath.suffix in ['.pkl', '.pt']:
                try:
                    agent, agent_type = load_agent(filepath)
                    agents_to_eval.append((filepath.stem, agent))
                except Exception as e:
                    print(f"Warning: Could not load {filepath.name}: {e}")
    elif args.agent:
        # Load specific agent type
        save_dir = Path(args.save_dir)
        
        if args.agent == 'montecarlo':
            # Try new naming first, fall back to old naming
            filepath = save_dir / 'monte_carlo_model.pkl'
            if not filepath.exists():
                filepath = save_dir / 'monte_carlo_agent.pkl'
        elif args.agent == 'qlearning':
            # Try new naming first, fall back to old naming
            filepath = save_dir / 'q_learning_model.pkl'
            if not filepath.exists():
                filepath = save_dir / 'q_learning_agent.pkl'
        elif args.agent == 'actorcritic':
            filepath = save_dir / 'actor_critic_best_model.pt'
        
        agent, agent_type = load_agent(filepath)
        agents_to_eval.append((args.agent, agent))
    else:
        print("Error: Must specify --agent, --agent-path, or --all")
        return
    
    # Evaluate agents
    print(f"\n{'='*60}")
    print(f"Evaluating {len(agents_to_eval)} agent(s) over {args.episodes} episodes")
    print(f"{'='*60}")
    
    all_results = {}
    
    for agent_name, agent in agents_to_eval:
        print(f"\nEvaluating {agent_name}...")
        results = evaluate_agent(env, agent, args.episodes, args.verbose)
        all_results[agent_name] = results
        print_results(agent_name, results)
    
    # Compare agents if multiple
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Agent':<25} {'Score':<20} {'Config Cost':<15} {'Total Reward':<15}")
        print(f"{'-'*60}")
        
        for agent_name, results in sorted(all_results.items(), 
                                         key=lambda x: x[1]['mean_score'], 
                                         reverse=True):
            score_str = f"{results['mean_score']:.1f} ± {results['std_score']:.1f}"
            print(f"{agent_name:<25} {score_str:<20} "
                  f"{results['mean_config_cost']:<15.1f} "
                  f"{results['total_reward']:<15,.0f}")
        print()


if __name__ == '__main__':
    main()
