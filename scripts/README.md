# RL Training Scripts

This directory contains modular Python scripts for training and evaluating RL agents on the Train Game environment.

## Overview

The codebase is organized into focused modules:

- **`agents.py`**: Agent implementations (Monte Carlo, Q-Learning, Actor-Critic)
- **`config.py`**: Configuration constants and hyperparameters
- **`train.py`**: Training script with CLI
- **`evaluate.py`**: Evaluation script for trained agents
- **`play_demo.py`**: Interactive demo viewer (terminal-based)
- **`play_demo_gui.py`**: Visual GUI demo launcher (recommended for demonstrations)
- **`visualize.py`**: Generate training curves and comparison plots
- **`viz_guide.py`**: Visualization workflow guide and examples

## Quick Start

### 1. Train an Agent

Train Q-Learning (recommended):
```bash
python train.py --agent qlearning --episodes 1000
```

Train all agents:
```bash
python train.py --agent all --episodes 1000
```

Train with custom settings:
```bash
python train.py --agent actorcritic --episodes 2000 --seed 123 --save-dir ./my_models
```

### 2. Evaluate Performance

Evaluate Q-Learning agent:
```bash
python evaluate.py --agent qlearning --episodes 100
```

Evaluate all saved agents:
```bash
python evaluate.py --all --episodes 100
```

Evaluate specific agent file:
```bash
python evaluate.py --agent-path ../saved_agents/q_learning_agent.pkl
```

### 3. Watch Agent Play

**Option A: Visual GUI Demo (Recommended)**
```bash
python play_demo_gui.py
```
Then click **"Agent Play"** and choose your agent. The GUI shows:
- Real-time train movement animation
- Live statistics (capacity, passengers, score)
- Action history log
- Visual station progression

**Option B: Terminal-Based Demo**
```bash
python play_demo.py --agent qlearning
```

Watch multiple episodes:
```bash
python play_demo.py --agent qlearning --episodes 5
```

Quiet mode (summary only):
```bash
python play_demo.py --agent qlearning --episodes 10 --quiet
```

### 4. Visualize Training Results 📊

**Generate comparison plots:**
```bash
python visualize.py --mode comparison
```
Creates: `../visualizations/agent_comparison.png`
- Score comparisons with error bars
- Action distribution breakdowns
- Performance consistency analysis

**Generate training curves:**
```bash
python visualize.py --mode training
```
Creates: `../visualizations/training_curves.png`
- Learning curves over time
- Cost and reward progression
- Efficiency metrics

**Generate everything:**
```bash
python visualize.py --mode all
```

**View visualization guide:**
```bash
python viz_guide.py
```

## Command Reference

### train.py

```
python train.py [OPTIONS]

Options:
  --agent {montecarlo,qlearning,actorcritic,all}
                        Which agent to train (default: qlearning)
  --episodes INT        Number of training episodes (default: 1000)
  --seed INT            Random seed for reproducibility (default: 42)
  --save-dir PATH       Directory to save trained agents (default: ../saved_agents)
  --no-save             Don't save trained agents
  --quiet               Suppress training progress output
```

**Examples:**
```bash
# Train Q-Learning with 1000 episodes
python train.py --agent qlearning --episodes 1000

# Train all agents without saving
python train.py --agent all --no-save

# Train Actor-Critic with custom seed
python train.py --agent actorcritic --seed 999
```

### evaluate.py

```
python evaluate.py [OPTIONS]

Options:
  --agent {montecarlo,qlearning,actorcritic}
                        Which agent to evaluate
  --agent-path PATH     Path to specific agent file
  --all                 Evaluate all saved agents
  --episodes INT        Number of evaluation episodes (default: 100)
  --save-dir PATH       Directory containing saved agents
  --verbose             Print episode-by-episode results
```

**Examples:**
```bash
# Quick evaluation of Q-Learning
python evaluate.py --agent qlearning --episodes 100

# Compare all agents
python evaluate.py --all

# Detailed evaluation with per-episode output
python evaluate.py --agent actorcritic --verbose
```

### play_demo.py

```
python play_demo.py [OPTIONS]

Options:
  --agent {montecarlo,qlearning,actorcritic}
                        Which agent to use (default: qlearning)
  --agent-path PATH     Path to specific agent file
  --episodes INT        Number of episodes to play (default: 1)
  --save-dir PATH       Directory containing saved agents
  --quiet               Only show final results
```

**Examples:**
```bash
# Watch one episode with Q-Learning
python play_demo.py --agent qlearning

# Watch 5 episodes with Actor-Critic
python play_demo.py --agent actorcritic --episodes 5

# Quick summary mode
python play_demo.py --agent qlearning --episodes 10 --quiet
```

## Project Structure

```
scripts/
├── agents.py           # Agent implementations + state processing
├── config.py           # Hyperparameters and constants
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation pipeline
├── play_demo.py        # Interactive demo
└── README.md           # This file

../
├── train_game_env.py   # Gymnasium environment
└── saved_agents/       # Trained model files
    ├── q_learning_agent.pkl
    ├── monte_carlo_agent.pkl
    └── actor_critic_best_model.pt
```

## Agent Details

### Q-Learning (Recommended)
- **Best Performance**: 93.6 ± 3.2 average score
- **File**: `q_learning_agent.pkl`
- **Method**: Temporal difference learning with experience replay
- **Hyperparameters**: α=0.1, γ=0.99, ε=0.1 → 0.01

### Actor-Critic
- **Performance**: 84.3 ± 10.2 average score
- **File**: `actor_critic_best_model.pt`
- **Method**: Policy gradient with value baseline
- **Hyperparameters**: lr=3e-4, γ=0.99, entropy=0.01

### Monte Carlo
- **Performance**: 33.5 ± 39.9 average score
- **File**: `monte_carlo_agent.pkl`
- **Method**: First-visit Monte Carlo
- **Hyperparameters**: γ=0.99, ε=0.1 → 0.01

## Modifying Hyperparameters

Edit `config.py` to change default hyperparameters:

```python
# Example: Make Q-Learning more exploratory
Q_LEARNING_CONFIG = {
    'alpha': 0.2,           # Higher learning rate
    'gamma': 0.99,
    'eps': 0.3,             # More exploration
    'eps_decay': 0.99,      # Slower decay
    'eps_min': 0.05,        # Higher minimum
    'alpha_decay': 0.9999,
    'alpha_min': 0.01
}
```

## Environment Details

- **Action Space**: Discrete(3)
  - 0: Add Carriage (+100 capacity, -10 cost)
  - 1: Widen Doors (+50 capacity, -5 cost)
  - 2: No Action (0 capacity, 0 cost)

- **State Space**: 6D continuous
  - [capacity, onboard, station_idx, direction, hour, minute]

- **Reward Function**: `(boarded × 1.5) - (unused × 0.5) - config_cost`

## Performance Tips

1. **Q-Learning is recommended** for this problem (best results)
2. **Train for 1000+ episodes** for convergence
3. **Use seed for reproducibility** (`--seed 42`)
4. **Evaluate over 100+ episodes** for stable metrics
5. **Actor-Critic needs longer training** (2000+ episodes recommended)

## Troubleshooting

**Issue**: Import errors
- **Solution**: Make sure you're in the `scripts/` directory when running
- **Solution**: The scripts auto-add parent directory to path for `train_game_env`

**Issue**: Agent file not found
- **Solution**: Train agents first with `train.py`
- **Solution**: Check `--save-dir` matches training directory

**Issue**: PyTorch not found
- **Solution**: Install PyTorch: `conda install pytorch` (in pt1 environment)

**Issue**: Poor performance
- **Solution**: Train longer (more episodes)
- **Solution**: Try Q-Learning instead of Monte Carlo
- **Solution**: Check hyperparameters in `config.py`

## Next Steps

1. **Train your agents**: Start with Q-Learning
2. **Evaluate performance**: Compare all agents
3. **Watch demos**: Understand agent behavior
4. **Modify game balance**: See `../GAME_BALANCE_GUIDE.md` for options
5. **Retrain with changes**: Experiment with different reward functions

## Related Files

- **`../GAME_BALANCE_GUIDE.md`**: Options to increase action variety
- **`../ACTION_SPACE_FIX.md`**: Documentation of 3-action system fix
- **`../rl_training.ipynb`**: Original notebook (kept for exploration)
