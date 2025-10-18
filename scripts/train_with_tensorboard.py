"""
Training script with TensorBoard logging for real-time monitoring.

Usage:
    python train_with_tensorboard.py --agent all --episodes 1000
    
    # In another terminal, start TensorBoard:
    tensorboard --logdir=../runs
    
    # Then open: http://localhost:6006
"""

import sys
import os
import argparse
import pickle
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
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


def train_monte_carlo_tb(env, episodes, writer, verbose=True):
    """Train Monte Carlo agent with TensorBoard logging."""
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
        
        # TensorBoard logging
        writer.add_scalar('MonteCarlo/Score', final_score, ep)
        writer.add_scalar('MonteCarlo/ConfigCost', info['total_config_cost'], ep)
        writer.add_scalar('MonteCarlo/EpisodeReward', ep_reward, ep)
        writer.add_scalar('MonteCarlo/Epsilon', agent.eps, ep)
        
        # Log moving averages
        if ep >= 50:
            writer.add_scalar('MonteCarlo/Score_MA50', np.mean(scores[-50:]), ep)
            writer.add_scalar('MonteCarlo/Score_Std', np.std(scores[-50:]), ep)
        
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


def train_qlearning_tb(env, episodes, writer, verbose=True):
    """Train Q-Learning agent with TensorBoard logging."""
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
        
        # TensorBoard logging
        writer.add_scalar('QLearning/Score', final_score, ep)
        writer.add_scalar('QLearning/ConfigCost', info['total_config_cost'], ep)
        writer.add_scalar('QLearning/EpisodeReward', ep_reward, ep)
        writer.add_scalar('QLearning/Epsilon', agent.eps, ep)
        writer.add_scalar('QLearning/Alpha', agent.alpha, ep)
        
        # Log moving averages
        if ep >= 50:
            writer.add_scalar('QLearning/Score_MA50', np.mean(scores[-50:]), ep)
            writer.add_scalar('QLearning/Score_Std', np.std(scores[-50:]), ep)
        
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


def train_actor_critic_tb(env, episodes, writer, verbose=True):
    """Train Actor-Critic agent with TensorBoard logging."""
    agent = ActorCriticAgent(**config.ACTOR_CRITIC_CONFIG)
    
    scores = []
    config_costs = []
    total_rewards = []
    actor_losses = []
    critic_losses = []
    
    for ep in range(episodes):
        state, info = env.reset()
        trajectory = []
        ep_reward = 0
        ep_actor_loss = 0
        ep_critic_loss = 0
        
        while True:
            action, log_prob, value = agent.policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append((state, (action, log_prob, value), reward, 
                             next_state, terminated, truncated))
            
            ep_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Learn and capture losses
        losses = agent.learn(trajectory)
        if losses:
            ep_actor_loss = losses.get('actor_loss', 0)
            ep_critic_loss = losses.get('critic_loss', 0)
            actor_losses.append(ep_actor_loss)
            critic_losses.append(ep_critic_loss)
        
        # Calculate final score
        final_score, _ = env.final_score()
        scores.append(final_score)
        config_costs.append(info['total_config_cost'])
        total_rewards.append(ep_reward)
        
        # TensorBoard logging
        writer.add_scalar('ActorCritic/Score', final_score, ep)
        writer.add_scalar('ActorCritic/ConfigCost', info['total_config_cost'], ep)
        writer.add_scalar('ActorCritic/EpisodeReward', ep_reward, ep)
        
        if losses:
            writer.add_scalar('ActorCritic/ActorLoss', ep_actor_loss, ep)
            writer.add_scalar('ActorCritic/CriticLoss', ep_critic_loss, ep)
        
        # Log moving averages
        if ep >= 50:
            writer.add_scalar('ActorCritic/Score_MA50', np.mean(scores[-50:]), ep)
            writer.add_scalar('ActorCritic/Score_Std', np.std(scores[-50:]), ep)
        
        if verbose and (ep + 1) % 100 == 0:
            recent_scores = scores[-100:]
            print(f"[AC] Episode {ep+1}/{episodes} | "
                  f"Avg Score: {np.mean(recent_scores):.1f} ± {np.std(recent_scores):.1f}")
    
    return agent, {
        'scores': scores,
        'config_costs': config_costs,
        'total_rewards': total_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses
    }


def save_agent(agent, agent_name, save_dir):
    """Save trained agent to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(agent, ActorCriticAgent):
        filepath = save_dir / f"{agent_name}_model.pt"
        torch.save(agent.net.state_dict(), filepath)
    else:
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
    parser = argparse.ArgumentParser(description='Train RL agents with TensorBoard')
    parser.add_argument('--agent', type=str, default='qlearning',
                       choices=['montecarlo', 'qlearning', 'actorcritic', 'all'],
                       help='Which agent to train (default: qlearning)')
    parser.add_argument('--episodes', type=int, default=config.DEFAULT_EPISODES,
                       help=f'Number of training episodes (default: {config.DEFAULT_EPISODES})')
    parser.add_argument('--seed', type=int, default=config.DEFAULT_SEED,
                       help=f'Random seed (default: {config.DEFAULT_SEED})')
    parser.add_argument('--save-dir', type=str, default=config.SAVED_AGENTS_DIR,
                       help='Directory to save trained agents')
    parser.add_argument('--log-dir', type=str, default='../runs',
                       help='TensorBoard log directory')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save trained agents')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress training progress output')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seed(args.seed)
    
    # Create environment
    env = TrainGameEnv()
    
    # Create TensorBoard writer with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = Path(args.log_dir) / f"train_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    verbose = not args.quiet
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Agent(s): {args.agent}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    print(f"  Save Dir: {args.save_dir}")
    print(f"  TensorBoard: {log_dir}")
    print(f"{'='*60}")
    print(f"\n💡 Start TensorBoard with:")
    print(f"   tensorboard --logdir={args.log_dir}")
    print(f"   Then open: http://localhost:6006\n")
    
    # Training
    start_time = time.time()
    
    if args.agent in ['montecarlo', 'all']:
        print("\n🎲 Training Monte Carlo Agent...")
        mc_agent, mc_results = train_monte_carlo_tb(env, args.episodes, writer, verbose)
        results['montecarlo'] = mc_results
        if not args.no_save:
            save_agent(mc_agent, 'monte_carlo', args.save_dir)
    
    if args.agent in ['qlearning', 'all']:
        print("\n🧠 Training Q-Learning Agent...")
        q_agent, q_results = train_qlearning_tb(env, args.episodes, writer, verbose)
        results['qlearning'] = q_results
        if not args.no_save:
            save_agent(q_agent, 'q_learning', args.save_dir)
    
    if args.agent in ['actorcritic', 'all']:
        print("\n🎭 Training Actor-Critic Agent...")
        ac_agent, ac_results = train_actor_critic_tb(env, args.episodes, writer, verbose)
        results['actorcritic'] = ac_results
        if not args.no_save:
            save_agent(ac_agent, 'actor_critic_best', args.save_dir)
    
    elapsed_time = time.time() - start_time
    
    # Save training history
    if not args.no_save and results:
        save_training_history(results, args.save_dir)
    
    # Log comparison metrics to TensorBoard
    if len(results) > 1:
        for agent_name, agent_results in results.items():
            final_scores = agent_results['scores'][-100:]
            final_costs = agent_results['config_costs'][-100:]
            
            writer.add_scalar(f'Comparison/FinalScore_{agent_name}', np.mean(final_scores), 0)
            writer.add_scalar(f'Comparison/FinalCost_{agent_name}', np.mean(final_costs), 0)
            writer.add_scalar(f'Comparison/Consistency_{agent_name}', np.std(final_scores), 0)
    
    writer.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Complete! ({elapsed_time:.1f}s)")
    print(f"{'='*60}\n")
    
    for agent_name, agent_results in results.items():
        scores = agent_results['scores']
        costs = agent_results['config_costs']
        rewards = agent_results['total_rewards']
        
        final_scores = scores[-100:]
        final_costs = costs[-100:]
        final_rewards = rewards[-100:]
        
        print(f"{agent_name.upper()}:")
        print(f"  Final Score: {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}")
        print(f"  Config Cost: {np.mean(final_costs):.1f}")
        print(f"  Total Reward: {np.sum(final_rewards):,.0f}")
        print()
    
    print(f"📊 View results in TensorBoard:")
    print(f"   tensorboard --logdir={args.log_dir}")
    print(f"   http://localhost:6006\n")


if __name__ == '__main__':
    main()
