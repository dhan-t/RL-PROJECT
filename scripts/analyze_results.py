import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define paths
PARENT_DIR = Path(__file__).parent.parent
LOGS_FILE = PARENT_DIR / 'saved_agents' / 'training_logs.json'
VISUALIZATIONS_DIR = PARENT_DIR / 'visualizations'

# Ensure visualization directory exists
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

def load_data(filepath):
    """Load training logs from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: Log file not found at {filepath}. Please run training first.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. The file might be corrupted.")
        return None

def plot_learning_curves(data):
    """Plot scores and cumulative rewards over episodes."""
    print("Generating learning curve plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for agent, history in data.items():
        df = pd.DataFrame(history)
        
        # Rolling average for score provides a smoothed view of performance
        ax1.plot(df['episode'], df['score'].rolling(window=50).mean(), label=f'{agent} (MA-50)')
        
        # Cumulative reward shows the total reward accumulated over time
        ax2.plot(df['episode'], df['total_reward'].cumsum(), label=agent)

    ax1.set_title('Learning Curve: Score Over Episodes (Smoothed)')
    ax1.set_ylabel('Average Score (50-episode MA)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title('Cumulative Reward Over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    save_path = VISUALIZATIONS_DIR / 'learning_curves.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves to {save_path}")

def plot_score_distribution(data):
    """Plot the distribution of final scores for each agent using a violin plot."""
    print("Generating score distribution plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores_data = []
    for agent, history in data.items():
        for h in history:
            scores_data.append({'Agent': agent, 'Score': h['score']})
    
    df = pd.DataFrame(scores_data)
    
    sns.violinplot(x='Agent', y='Score', data=df, ax=ax, inner='quartile', cut=0)
    
    ax.set_title('Distribution of Final Scores per Agent')
    ax.set_ylabel('Final Score')
    ax.set_xlabel('Agent')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = VISUALIZATIONS_DIR / 'score_distribution.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved score distribution to {save_path}")

def plot_efficiency(data):
    """Plot score per unit of configuration cost."""
    print("Generating efficiency plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    efficiency_data = []
    for agent, history in data.items():
        for h in history:
            # Avoid division by zero; if cost is 0, efficiency is simply the score
            cost = h['config_cost']
            score = h['score']
            if cost > 0:
                efficiency = score / cost
            else:
                # If cost is zero, a high positive score is infinitely efficient,
                # a zero score has zero efficiency, and a negative score is infinitely inefficient.
                # We can cap large values, but representing as score is reasonable.
                efficiency = score
            efficiency_data.append({'Agent': agent, 'Efficiency': efficiency})
            
    df = pd.DataFrame(efficiency_data)
    
    sns.boxplot(x='Agent', y='Efficiency', data=df, ax=ax)
    
    ax.set_title('Agent Efficiency (Score per Unit Cost)')
    ax.set_ylabel('Efficiency (Score / Config Cost)')
    ax.set_xlabel('Agent')
    # --- BEST PRACTICE FIX ---
    # Removed `ax.set_ylim(bottom=0)`. This allows the plot to show negative
    # efficiency values, giving a more complete and honest view of performance,
    # especially for poorly performing agents.
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = VISUALIZATIONS_DIR / 'efficiency_comparison.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved efficiency comparison to {save_path}")

def plot_action_distribution(data):
    """Plot the distribution of actions taken by each agent."""
    print("Generating action distribution plots...")
    action_map = {0: 'Add Carriage', 1: 'Widen Carriage', 2: 'No Action'}
    
    agents = list(data.keys())
    fig, axes = plt.subplots(1, len(agents), figsize=(15, 5), sharey=True)
    if len(agents) == 1: # Ensure axes is always iterable
        axes = [axes]

    for ax, agent in zip(axes, agents):
        all_actions = []
        for h in data[agent]:
            for step in h['steps']:
                all_actions.append(step['action'])
        
        action_counts = pd.Series(all_actions).value_counts().sort_index()
        # Ensure all possible actions are present for consistent plotting
        action_counts = action_counts.reindex(action_map.keys(), fill_value=0)
        action_counts.index = action_counts.index.map(action_map)
        
        action_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'{agent} Actions')
        ax.tick_params(axis='x', rotation=45)
    
    axes[0].set_ylabel('Frequency')
    plt.suptitle('Action Distribution Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = VISUALIZATIONS_DIR / 'action_distribution.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved action distribution to {save_path}")

def plot_performance_summary(data):
    """Create a summary bar chart of key performance indicators."""
    print("Generating performance summary plot...")
    summary_data = []
    for agent, history in data.items():
        df = pd.DataFrame(history)
        avg_score = df['score'].mean()
        avg_cost = df['config_cost'].mean()
        
        # --- BEST PRACTICE NOTE ---
        # This metric can be misleading. An agent that consistently gets a score of 0
        # will have a low std dev and thus high "consistency". Always view this
        # plot alongside the score distribution (violin) plot for true insight.
        std_dev = df['score'].std()
        consistency = 1 / std_dev if std_dev > 0 else 0
        
        summary_data.append({
            'Agent': agent,
            'Average Final Score': avg_score,
            'Average Configuration Cost': avg_cost,
            'Performance Consistency (1/StdDev)': consistency
        })
        
    df_summary = pd.DataFrame(summary_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.barplot(x='Agent', y='Average Final Score', data=df_summary, ax=axes[0])
    axes[0].set_title('Average Final Score')
    
    sns.barplot(x='Agent', y='Average Configuration Cost', data=df_summary, ax=axes[1])
    axes[1].set_title('Average Configuration Cost')
    
    sns.barplot(x='Agent', y='Performance Consistency (1/StdDev)', data=df_summary, ax=axes[2])
    axes[2].set_title('Performance Consistency (1 / Std Dev)')
    
    plt.suptitle('Agent Performance Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = VISUALIZATIONS_DIR / 'performance_summary.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved performance summary to {save_path}")

def main():
    """Main function to load data and generate all plots."""
    training_data = load_data(LOGS_FILE)
    
    if training_data:
        # Set a clean, modern plot style
        sns.set_theme(style="whitegrid")
        
        # Generate all plots
        plot_learning_curves(training_data)
        plot_score_distribution(training_data)
        plot_efficiency(training_data)
        plot_action_distribution(training_data)
        plot_performance_summary(training_data)
        
        print("\nAll visualizations have been generated and saved in the 'visualizations' directory.")

if __name__ == '__main__':
    main()