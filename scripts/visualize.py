"""
Visualization script for analyzing training results and comparing agents.

Usage:
    # After training, visualize results
    python visualize.py --mode training
    
    # Compare agent performance
    python visualize.py --mode comparison
    
    # Show all visualizations
    python visualize.py --mode all
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from train_game_env import TrainGameEnv

from agents import MonteCarloAgent, QLearningAgent, ActorCriticAgent
import config


def load_training_history(save_dir):
    """Load training history if available."""
    history_file = Path(save_dir) / 'training_history.pkl'
    if history_file.exists():
        with open(history_file, 'rb') as f:
            return pickle.load(f)
    return None


def plot_training_curves(history, save_path=None):
    """Plot training curves for all agents."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    colors = {
        'montecarlo': '#FF6B6B',
        'qlearning': '#4ECDC4',
        'actorcritic': '#95E1D3'
    }
    
    # 1. Scores over time
    ax1 = fig.add_subplot(gs[0, :])
    for agent_name, data in history.items():
        scores = data['scores']
        episodes = range(1, len(scores) + 1)
        
        # Plot with smoothing
        window = 50
        if len(scores) >= window:
            smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(scores) + 1), smoothed, 
                    label=agent_name.title(), color=colors.get(agent_name, 'gray'), linewidth=2)
        else:
            ax1.plot(episodes, scores, label=agent_name.title(), 
                    color=colors.get(agent_name, 'gray'), alpha=0.3)
    
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Curves: Score Over Time (50-episode moving average)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Config costs over time
    ax2 = fig.add_subplot(gs[1, 0])
    for agent_name, data in history.items():
        costs = data['config_costs']
        episodes = range(1, len(costs) + 1)
        window = 50
        if len(costs) >= window:
            smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(costs) + 1), smoothed, 
                    label=agent_name.title(), color=colors.get(agent_name, 'gray'))
    
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Config Cost', fontsize=10)
    ax2.set_title('Configuration Costs', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Total rewards over time
    ax3 = fig.add_subplot(gs[1, 1])
    for agent_name, data in history.items():
        rewards = data['total_rewards']
        episodes = range(1, len(rewards) + 1)
        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax3.plot(range(window, len(rewards) + 1), smoothed, 
                    label=agent_name.title(), color=colors.get(agent_name, 'gray'))
    
    ax3.set_xlabel('Episode', fontsize=10)
    ax3.set_ylabel('Total Reward', fontsize=10)
    ax3.set_title('Cumulative Rewards', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Score distribution (final 200 episodes)
    ax4 = fig.add_subplot(gs[1, 2])
    final_scores = []
    labels = []
    for agent_name, data in history.items():
        scores = data['scores'][-200:]  # Last 200 episodes
        final_scores.append(scores)
        labels.append(agent_name.title())
    
    bp = ax4.boxplot(final_scores, labels=labels, patch_artist=True)
    for patch, agent_name in zip(bp['boxes'], history.keys()):
        patch.set_facecolor(colors.get(agent_name, 'gray'))
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_title('Score Distribution (Last 200 Episodes)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Learning progression (early vs late)
    ax5 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(history))
    width = 0.35
    
    early_scores = [np.mean(data['scores'][:100]) for data in history.values()]
    late_scores = [np.mean(data['scores'][-100:]) for data in history.values()]
    
    ax5.bar(x - width/2, early_scores, width, label='First 100 Episodes',
            color='lightcoral', alpha=0.8)
    ax5.bar(x + width/2, late_scores, width, label='Last 100 Episodes',
            color='lightgreen', alpha=0.8)
    
    ax5.set_xlabel('Agent', fontsize=10)
    ax5.set_ylabel('Average Score', fontsize=10)
    ax5.set_title('Learning Progress: Early vs Late', fontsize=11, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([name.title() for name in history.keys()], rotation=15)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Efficiency: Score per cost
    ax6 = fig.add_subplot(gs[2, 1])
    efficiency = []
    labels = []
    for agent_name, data in history.items():
        final_scores = data['scores'][-100:]
        final_costs = data['config_costs'][-100:]
        # Calculate efficiency (avoid division by zero)
        eff = [s / max(c, 0.1) for s, c in zip(final_scores, final_costs)]
        efficiency.append(np.mean(eff))
        labels.append(agent_name.title())
    
    bars = ax6.bar(labels, efficiency, color=[colors.get(name, 'gray') for name in history.keys()],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Score / Config Cost', fontsize=10)
    ax6.set_title('Efficiency: Score per Unit Cost', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Variance analysis
    ax7 = fig.add_subplot(gs[2, 2])
    variances = []
    labels = []
    for agent_name, data in history.items():
        scores = data['scores'][-100:]
        variances.append(np.std(scores))
        labels.append(agent_name.title())
    
    bars = ax7.bar(labels, variances, color=[colors.get(name, 'gray') for name in history.keys()],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Standard Deviation', fontsize=10)
    ax7.set_title('Performance Consistency', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('🚆 Training Analysis Dashboard', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_agent_comparison(save_dir, save_path=None):
    """Compare final performance of all agents with detailed metrics."""
    env = TrainGameEnv()
    
    # Load agents
    agents = {}
    agent_files = {
        'Monte Carlo': ['monte_carlo_model.pkl', 'monte_carlo_agent.pkl'],
        'Q-Learning': ['q_learning_model.pkl', 'q_learning_agent.pkl'],
        'Actor-Critic': ['actor_critic_best_model.pt']
    }
    
    save_dir = Path(save_dir)
    
    for name, possible_files in agent_files.items():
        for filename in possible_files:
            filepath = save_dir / filename
            if filepath.exists():
                try:
                    if filepath.suffix == '.pt':
                        import torch
                        agent = ActorCriticAgent(**config.ACTOR_CRITIC_CONFIG)
                        agent.net.load_state_dict(torch.load(filepath))
                        agent.net.eval()
                    else:
                        with open(filepath, 'rb') as f:
                            agent = pickle.load(f)
                    agents[name] = agent
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {name}: {e}")
    
    if not agents:
        print("❌ No trained agents found!")
        return
    
    # Evaluate agents
    print("🔍 Evaluating agents for comparison...")
    n_episodes = 100
    results = {}
    
    for name, agent in agents.items():
        print(f"  Evaluating {name}...")
        scores = []
        costs = []
        actions = {0: 0, 1: 0, 2: 0}
        
        for ep in range(n_episodes):
            state, info = env.reset()
            ep_cost = 0
            
            while True:
                # Get action
                if isinstance(agent, ActorCriticAgent):
                    action, _, _ = agent.policy(state, greedy=True)
                else:
                    action = agent.policy(state, greedy=True)
                
                actions[action] += 1
                state, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            final_score, _ = env.final_score()
            scores.append(final_score)
            costs.append(info['total_config_cost'])
        
        results[name] = {
            'scores': scores,
            'costs': costs,
            'actions': actions
        }
    
    # Create comparison plots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    colors = {'Monte Carlo': '#FF6B6B', 'Q-Learning': '#4ECDC4', 'Actor-Critic': '#95E1D3'}
    
    # 1. Score comparison
    ax1 = fig.add_subplot(gs[0, 0])
    agent_names = list(results.keys())
    mean_scores = [np.mean(results[name]['scores']) for name in agent_names]
    std_scores = [np.std(results[name]['scores']) for name in agent_names]
    
    bars = ax1.bar(agent_names, mean_scores, yerr=std_scores, capsize=10,
                   color=[colors[name] for name in agent_names], alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Average Score (±1 SD)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, mean_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Config cost comparison
    ax2 = fig.add_subplot(gs[0, 1])
    mean_costs = [np.mean(results[name]['costs']) for name in agent_names]
    
    bars = ax2.bar(agent_names, mean_costs,
                   color=[colors[name] for name in agent_names], alpha=0.8,
                   edgecolor='black', linewidth=2)
    ax2.set_ylabel('Config Cost', fontsize=12, fontweight='bold')
    ax2.set_title('Average Configuration Cost', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, cost in zip(bars, mean_costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Score distribution violin plot
    ax3 = fig.add_subplot(gs[0, 2])
    score_data = [results[name]['scores'] for name in agent_names]
    parts = ax3.violinplot(score_data, positions=range(len(agent_names)), 
                           showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[agent_names[i]])
        pc.set_alpha(0.7)
    
    ax3.set_xticks(range(len(agent_names)))
    ax3.set_xticklabels(agent_names, rotation=15)
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Action distribution
    ax4 = fig.add_subplot(gs[1, :])
    action_names = ['Add Carriage', 'Widen Carriage', 'No Action']
    x = np.arange(len(agent_names))
    width = 0.25
    
    action_colors = ['#FFB6C1', '#87CEEB', '#98FB98']
    
    for i in range(3):
        action_counts = [results[name]['actions'][i] for name in agent_names]
        total_actions = [sum(results[name]['actions'].values()) for name in agent_names]
        percentages = [(count / total * 100) for count, total in zip(action_counts, total_actions)]
        
        bars = ax4.bar(x + i * width, percentages, width, label=action_names[i],
                      color=action_colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            if height > 2:  # Only show label if bar is visible
                ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{pct:.0f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=9, color='black')
    
    ax4.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Action Usage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Action Distribution Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(agent_names)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🏆 Agent Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 PERFORMANCE SUMMARY")
    print("="*70)
    for name in agent_names:
        scores = results[name]['scores']
        costs = results[name]['costs']
        print(f"\n{name}:")
        print(f"  Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"  Config Cost: {np.mean(costs):.1f}")
        print(f"  Consistency (lower is better): {np.std(scores):.1f}")
        print(f"  Efficiency (score/cost): {np.mean(scores)/max(np.mean(costs), 0.1):.2f}")
        
        total_actions = sum(results[name]['actions'].values())
        print(f"  Action Distribution:")
        for action_idx, action_name in enumerate(action_names):
            count = results[name]['actions'][action_idx]
            pct = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"    {action_name}: {pct:.1f}%")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--mode', type=str, default='comparison',
                       choices=['training', 'comparison', 'all'],
                       help='Visualization mode (default: comparison)')
    parser.add_argument('--save-dir', type=str, default=config.SAVED_AGENTS_DIR,
                       help='Directory containing saved agents')
    parser.add_argument('--output-dir', type=str, default='../visualizations',
                       help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots instead of saving')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🎨 VISUALIZATION GENERATOR")
    print(f"{'='*70}\n")
    
    if args.mode in ['training', 'all']:
        print("📈 Generating training curves...")
        history = load_training_history(args.save_dir)
        
        if history:
            save_path = None if args.show else output_dir / 'training_curves.png'
            plot_training_curves(history, save_path)
        else:
            print("⚠️ No training history found. Train with history tracking first.")
    
    if args.mode in ['comparison', 'all']:
        print("📊 Generating agent comparison...")
        save_path = None if args.show else output_dir / 'agent_comparison.png'
        plot_agent_comparison(args.save_dir, save_path)
    
    print(f"\n✅ Visualization complete!")
    if not args.show:
        print(f"📁 Plots saved to: {output_dir}/")


if __name__ == '__main__':
    main()
