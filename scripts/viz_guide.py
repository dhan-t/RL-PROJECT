"""
Quick visualization examples and usage guide.

This script demonstrates common visualization workflows.
"""

import subprocess
import sys

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def run_visualization_workflow():
    """Complete workflow: train → visualize → compare"""
    
    print_header("🎨 VISUALIZATION WORKFLOW GUIDE")
    
    print("This guide shows you how to generate beautiful plots for your RL agents.\n")
    
    # Step 1
    print("STEP 1: Train agents (if not already done)")
    print("-" * 70)
    print("Command:")
    print("  python train.py --agent all --episodes 1000\n")
    print("This saves:")
    print("  • Trained agent models (.pkl and .pt files)")
    print("  • Training history (training_history.pkl)")
    print()
    
    # Step 2
    print("STEP 2: Generate comparison plots")
    print("-" * 70)
    print("Command:")
    print("  python visualize.py --mode comparison\n")
    print("This creates:")
    print("  • Agent performance comparison")
    print("  • Score distributions")
    print("  • Action usage breakdown")
    print("  • Saved to: ../visualizations/agent_comparison.png")
    print()
    
    # Step 3
    print("STEP 3: View training curves (if history available)")
    print("-" * 70)
    print("Command:")
    print("  python visualize.py --mode training\n")
    print("This creates:")
    print("  • Learning curves over time")
    print("  • Cost and reward progression")
    print("  • Efficiency analysis")
    print("  • Saved to: ../visualizations/training_curves.png")
    print()
    
    # Step 4
    print("STEP 4: Generate all visualizations")
    print("-" * 70)
    print("Command:")
    print("  python visualize.py --mode all\n")
    print("Creates both training curves and comparison plots!")
    print()
    
    # Additional options
    print("ADVANCED OPTIONS")
    print("-" * 70)
    print("Show plots interactively (don't save):")
    print("  python visualize.py --mode comparison --show")
    print()
    print("Custom output directory:")
    print("  python visualize.py --mode all --output-dir ./my_plots")
    print()
    print("Custom save directory for agents:")
    print("  python visualize.py --save-dir ./my_models")
    print()
    
    # Quick examples
    print_header("🚀 QUICK EXAMPLES")
    
    examples = [
        ("Compare all trained agents", "python visualize.py --mode comparison"),
        ("View training progress", "python visualize.py --mode training"),
        ("Generate everything", "python visualize.py --mode all"),
        ("Interactive plot viewer", "python visualize.py --mode comparison --show"),
    ]
    
    for description, command in examples:
        print(f"• {description}")
        print(f"  $ {command}\n")
    
    # What each plot shows
    print_header("📊 WHAT EACH PLOT SHOWS")
    
    print("TRAINING CURVES (--mode training):")
    print("  1. Learning curves: Score improvement over episodes")
    print("  2. Config costs: How spending changes during learning")
    print("  3. Rewards: Cumulative reward progression")
    print("  4. Score distribution: Consistency analysis")
    print("  5. Early vs Late: Learning progress comparison")
    print("  6. Efficiency: Score per unit cost")
    print("  7. Variance: Performance stability\n")
    
    print("AGENT COMPARISON (--mode comparison):")
    print("  1. Average scores: Which agent performs best")
    print("  2. Config costs: Which agent is most economical")
    print("  3. Score distribution: Performance consistency")
    print("  4. Action distribution: Strategy differences\n")
    
    print_header("💡 TIPS")
    
    tips = [
        "Always train with --agent all to compare all three algorithms",
        "Use --episodes 1000 or more for meaningful learning curves",
        "Check action distribution to understand agent strategies",
        "Low variance = consistent agent, high variance = exploratory",
        "Score/cost efficiency shows which agent is most economical",
        "Actor-Critic usually converges to extreme strategies (all or nothing)",
        "Q-Learning tends to have moderate action diversity",
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    
    print("\n" + "="*70)
    print("Ready to visualize? Run: python visualize.py --mode all")
    print("="*70 + "\n")

if __name__ == '__main__':
    run_visualization_workflow()
