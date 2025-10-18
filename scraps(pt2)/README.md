# 🚆 Train Capacity Management with Reinforcement Learning# RL PROJECT: Train Capacity Management Game



**A comprehensive RL project demonstrating Monte Carlo, Q-Learning, and Actor-Critic algorithms on a real-world transit optimization problem.**A Gymnasium-compliant reinforcement learning environment for training agents to manage train capacity in a realistic transit system simulation.



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)## 🎯 Overview

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26+-green.svg)](https://gymnasium.farama.org/)This project implements a train capacity management game based on Manila's LRT-2 line, where RL agents learn to balance:

- **Passenger boarding** (maximize)

---- **Capacity efficiency** (minimize waste)

- **Infrastructure stress** (prevent collapse)

## 📋 Table of Contents- **Configuration costs** (minimize expansions)



- [Overview](#-overview)## ✅ Fixed Issues (Comprehensive Review)

- [Key Results](#-key-results)

- [Quick Start](#-quick-start)### **Critical Fixes Applied:**

- [Project Structure](#-project-structure)

- [Documentation](#-documentation)1. **✅ Gymnasium API Compliance**

- [Visualization](#-visualization)   - `step()` now returns 5-tuple: `(obs, reward, terminated, truncated, info)`

- [TensorBoard](#-tensorboard-integration)   - `reset()` now returns 2-tuple: `(obs, info)`

- [Future Work](#-future-work)   - Action and observation spaces are attributes (not properties)

   - Added proper `render_mode` support

---

2. **✅ Reward Double-Counting Bug Fixed**

## 🎯 Overview   - Configuration cost was being subtracted twice

   - Now applied only once in `step_reward` calculation

This project implements a **Gymnasium-compliant environment** simulating Manila's LRT-2 train system, where RL agents learn optimal capacity management strategies across 13 stations with dynamic passenger demand.

3. **✅ Consistent State Space**

### Environment Features   - 6-dimensional state: `[capacity, passengers, station_idx, direction, hour, minute]`

   - All files updated to use the same dimensions

- **State Space:** 6D continuous (capacity, passengers, station, direction, time)

- **Action Space:** 3 discrete actions (Add Carriage +100, Widen +50, No Action)4. **✅ Info Dictionary Consistency**

- **Reward Structure:** Balance boarding, efficiency, costs, and penalties   - Standardized structure across all code paths

- **Realistic Simulation:** Time-of-day multipliers, station types, capacity decay   - Includes: alighted, boarded, arrivals, penalties, efficiency metrics



### Algorithms Implemented5. **✅ Agent Implementations Updated**

   - Fixed Gymnasium API compatibility in all training loops

1. **Monte Carlo** - First-visit MC with epsilon-greedy exploration   - Updated Q-Learning, Monte Carlo, and Actor-Critic agents

2. **Q-Learning** - Temporal-difference learning with decay schedules

3. **Actor-Critic** - Policy gradient with baseline (PyTorch neural network)## 🚀 Quick Start



---### Prerequisites



## 🏆 Key Results```bash

# Activate the conda environment with gymnasium

**Training:** 1000 episodes per agent | **Evaluation:** 100 episodes | **Seed:** 42conda activate pt1

```

| Agent | Score | Consistency | Efficiency | Strategy |

|-------|-------|-------------|------------|----------|### Basic Usage

| **Actor-Critic** 🥇 | **56.9 ± 4.0** | ⭐⭐⭐⭐⭐ | **568.70** | 100% No Action |

| **Q-Learning** 🥈 | **55.4 ± 42.3** | ⭐⭐ | **1.13** | 71% No Action |```python

| **Monte Carlo** 🥉 | **41.6 ± 43.1** | ⭐ | **0.68** | 61% No Action |from train_game_env import TrainGameEnv



### Key Findings# Create environment

env = TrainGameEnv(initial_capacity=100, seed=42)

✅ **Actor-Critic achieved optimal convergence** - discovered "No Action" as mathematically optimal  

✅ **Policy gradients converge faster** - 600 episodes vs 800+ for value-based methods  # Reset environment

✅ **Low variance indicates convergence** - Actor-Critic ±4.0 vs Q-Learning ±42.3  obs, info = env.reset()

✅ **Game design impacts strategy** - current parameters favor conservative play  

# Take a step

📄 **Detailed analysis:** [ANALYSIS_REPORT.md](./ANALYSIS_REPORT.md)action = env.action_space.sample()  # or 0, 1, 2

obs, reward, terminated, truncated, info = env.step(action)

---

# Get final score

## 🚀 Quick Startif terminated or truncated:

    normalized_score, raw_score = env.final_score()

### Installation    print(f"Final Score: {normalized_score}/100")

```

```bash

# Clone repository## 📊 Environment Specification

git clone https://github.com/yourusername/RL-PROJECT.git

cd RL-PROJECT### Action Space

- **Type:** `Discrete(3)`

# Activate conda environment- **Actions:**

conda activate pt1  - `0`: Add carriage (+100 capacity, cost=10, weight=1.0)

  - `1`: Widen carriage (+50 capacity, cost=5, weight=0.5)

# Install dependencies (if needed)  - `2`: No action (cost=0, weight=0)

conda install pytorch numpy matplotlib gymnasium

```### Observation Space

- **Type:** `Box(6,)` with dtype `float32`

### Train Agents- **Dimensions:**

  1. `capacity`: Current train capacity [0, ∞)

```bash  2. `passengers_onboard`: Current passengers [0, ∞)

cd scripts  3. `station_idx`: Current station [0, 12]

  4. `direction`: Travel direction {-1, 1}

# Train all agents (recommended)  5. `hour`: Current hour [0, 23]

python train.py --agent all --episodes 1000  6. `minute`: Current minute [0, 59]



# Train specific agent### Reward Structure

python train.py --agent qlearning --episodes 1000

```python

# Train with TensorBoard loggingreward = boarding_reward - efficiency_penalty - config_penalty

python train_with_tensorboard.py --agent all --episodes 1000

```where:

  boarding_reward = 1.5 × passengers_boarded

### Evaluate Performance  efficiency_penalty = calculated based on unused capacity and time

  config_penalty = 2.0 × action_cost

```bash```

# Evaluate all agents

python evaluate.py --all --episodes 100### Termination Conditions

- **Terminated:** Infrastructure collapse or end of operating hours (22:00)

# Evaluate specific agent- **Truncated:** Maximum steps reached (2000)

python evaluate.py --agent actorcritic --episodes 100

```## 🤖 Training RL Agents



### Visualize ResultsThe project includes three RL algorithms:



```bash1. **Monte Carlo** (tabular, first-visit)

# Generate all plots2. **Q-Learning** (tabular, off-policy)

python visualize.py --mode all3. **Actor-Critic** (neural network, policy gradient)



# Agent comparison only### Running Training

python visualize.py --mode comparison

Open `rl_training.ipynb` in Jupyter:

# Training curves only

python visualize.py --mode training```bash

conda activate pt1

# Interactive mode (show plots)jupyter notebook rl_training.ipynb

python visualize.py --mode all --show```

```

Or run cells directly to train agents.

### Watch Agents Play

## 📁 Project Structure

```bash

# GUI demo (recommended)```

python play_demo_gui.pyRL-PROJECT/

├── train_game_env.py       # Main Gymnasium environment

# Terminal demo├── rl_training.ipynb        # Agent training notebook

python play_demo.py --agent actorcritic --episodes 3├── GAME.ipynb               # Game development history

```├── SCRAP_NB.ipynb          # Experimental code

├── ISSUES_AND_FIXES.md     # Comprehensive issue analysis

---└── README.md               # This file

```

## 📁 Project Structure

## 🔧 Key Design Decisions

```

RL-PROJECT/### Why These Fixes Matter

├── 📄 ANALYSIS_REPORT.md          # Comprehensive results analysis

├── 📄 REBALANCING_GUIDE.md        # Future game design improvements1. **Gymnasium Compliance**: Enables use with Stable-Baselines3, RLlib, and other standard RL libraries

├── 📄 VISUALIZATION_GUIDE.md      # Plotting quickstart2. **Reward Fixes**: Ensures agents can actually learn (previous double-penalty made learning nearly impossible)

├── 📄 GAME_BALANCE_GUIDE.md       # Game mechanics documentation3. **State Consistency**: Prevents crashes and confusion across different code versions

│4. **Proper API**: Follows RL community standards for environment design

├── scripts/                        # Main codebase

│   ├── agents.py                   # RL algorithm implementations### Reward Design Philosophy

│   ├── config.py                   # Hyperparameters & constants

│   ├── train.py                    # Training pipelineThe reward structure balances:

│   ├── train_with_tensorboard.py   # Training with TB logging- **Positive reinforcement** for serving passengers

│   ├── evaluate.py                 # Evaluation framework- **Negative feedback** for inefficiency (unused capacity)

│   ├── visualize.py                # Matplotlib plotting- **Cost-awareness** for infrastructure expansion

│   ├── play_demo.py                # Terminal demo- **Time-varying difficulty** to prevent exploitation

│   ├── play_demo_gui.py            # GUI launcher

│   ├── viz_guide.py                # Visualization guide## 📈 Expected Performance

│   └── README.md                   # Scripts documentation

│With proper training (500+ episodes):

├── train_game_env.py               # Gymnasium environment- **Random Agent:** ~20-30/100

├── gui_train_game.py               # Standalone GUI application- **Monte Carlo:** ~40-55/100

├── test_3action_system.py          # Environment tests- **Q-Learning:** ~45-60/100

│- **Actor-Critic:** ~50-70/100 (best)

├── saved_agents/                   # Trained model checkpoints

│   ├── monte_carlo_model.pkl## 🐛 Known Limitations

│   ├── q_learning_model.pkl

│   └── actor_critic_best_model.pt1. **State space explosion** for tabular methods (discretization needed)

│2. **Long training times** for Actor-Critic (neural network)

├── visualizations/                 # Generated plots3. **Sparse reward signal** in some scenarios

│   ├── training_curves.png4. **Stochastic environment** makes learning challenging

│   └── agent_comparison.png

│## 🔬 Future Improvements

└── runs/                           # TensorBoard logs

    └── train_YYYYMMDD-HHMMSS/- [ ] Add reward normalization wrapper

```- [ ] Implement PPO/SAC for better performance

- [ ] Add action masking for invalid actions

---- [ ] Create vectorized environment for parallel training

- [ ] Add more sophisticated state discretization

## 📚 Documentation- [ ] Implement curriculum learning



### Core Documents## 📚 References



- **[ANALYSIS_REPORT.md](./ANALYSIS_REPORT.md)** - Complete results, insights, and research findings- **Gymnasium Documentation:** https://gymnasium.farama.org/

- **[REBALANCING_GUIDE.md](./REBALANCING_GUIDE.md)** - How to modify game for more complex strategies- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/

- **[VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md)** - Plotting quick start and customization- **Sutton & Barto:** Reinforcement Learning: An Introduction

- **[scripts/README.md](./scripts/README.md)** - API reference and command guide

## 🤝 Contributing

### Guides

This is an educational project. Feel free to experiment with:

- **[GAME_BALANCE_GUIDE.md](./GAME_BALANCE_GUIDE.md)** - Game mechanics and parameter tuning- Different reward structures

- **[RETRAINING_GUIDE.md](./RETRAINING_GUIDE.md)** - How to retrain after changes- New RL algorithms

- Environment modifications

---- Better state representations



## 📊 Visualization## 📝 License



### Generated PlotsEducational use only.



**Training Curves Dashboard** (`training_curves.png`):---

- Learning curves (50-episode moving average)

- Config costs over time**Last Updated:** October 11, 2025  

- Reward progression**Status:** ✅ All critical issues fixed and tested  

- Score distributions**Environment:** Fully Gymnasium-compliant

- Early vs late performance

- Efficiency metrics (score/cost)

- Variance analysis

A Gymnasium-compatible reinforcement learning environment for train capacity optimization, simulating the LRT-2 line in Metro Manila.

**Agent Comparison** (`agent_comparison.png`):

- Average scores with error bars## 🎯 Overview

- Configuration cost comparison

- Score distribution violin plotsThis project implements a realistic train capacity management simulation where RL agents must balance:

- Action usage breakdown (stacked bars)- **Passenger boarding** (reward for serving passengers)

- **Capacity efficiency** (penalty for unused space)

### Generate Plots- **Infrastructure stress** (risk of collapse from overbuilding)

- **Configuration costs** (penalty for capacity changes)

```bash

# Quick generation## 🏗️ Project Structure

python scripts/visualize.py --mode all

```

# Custom output directoryRL-PROJECT/

python scripts/visualize.py --mode all --output-dir ./my_plots├── train_game_env.py      # Gymnasium-compliant environment

├── rl_training.ipynb      # Agent implementations & training

# Interactive viewer├── test_environment.py    # Unit tests for environment

python scripts/visualize.py --mode comparison --show├── ISSUES_AND_FIXES.md    # Detailed analysis of bugs found

├── GAME.ipynb             # Legacy versions (for reference)

# View guide└── README.md              # This file

python scripts/viz_guide.py```

```

## 🚀 Quick Start

---

### Installation

## 🔥 TensorBoard Integration

```bash

### Real-Time Monitoring# Install dependencies

pip install gymnasium torch matplotlib numpy

```bash```

# Terminal 1: Train with TensorBoard logging

cd scripts### Running Tests

python train_with_tensorboard.py --agent all --episodes 1000

```bash

# Terminal 2: Start TensorBoard# Test environment correctness

tensorboard --logdir=../runspython test_environment.py

```

# Open browser: http://localhost:6006

```### Training Agents



### What TensorBoard ShowsOpen `rl_training.ipynb` in Jupyter and run all cells to:

1. Test environment compatibility

**Scalars:**2. Load improved agent implementations

- Episode scores (per agent)3. Train Monte Carlo, Q-Learning, and Actor-Critic agents

- Configuration costs4. Compare performance with visualizations

- Episode rewards

- Exploration parameters (epsilon, alpha)## 🎮 Environment Details

- Loss functions (Actor-Critic)

- Moving averages (50-episode window)### State Space (6D)

- Standard deviations- `capacity`: Current train capacity [0, ∞)

- Comparison metrics- `passengers_onboard`: Current passenger count [0, capacity]

- `station_idx`: Current station [0-12]

**Graphs:**- `direction`: Travel direction {-1, 1}

- Network architecture (Actor-Critic)- `hour`: Current hour [0-23]

- Computational graph- `minute`: Current minute [0-59]



**Projector:**### Action Space (Discrete, 3 actions)

- State space embeddings (optional)- `0`: Add carriage (+100 capacity, cost=10, weight=1.0)

- `1`: Widen carriage (+50 capacity, cost=5, weight=0.5)

---- `2`: No action (cost=0, weight=0)



## 🔬 Research Insights### Reward Structure

```python

### Exploration vs Exploitationreward = (1.5 × boarded) - efficiency_penalty - (2.0 × cost)

```

```

Monte Carlo:  High exploration (ε=0.01) → Slow convergence → High variance### Episode Termination

Q-Learning:   Balanced (ε-greedy)      → Moderate speed  → Moderate variance  - **Terminated**: Infrastructure collapse or operating hours end (22:00)

Actor-Critic: Fast exploitation        → Fast convergence → Low variance- **Truncated**: Maximum steps reached (2000)

```

## 🤖 Agent Implementations

### Convergence Patterns

### 1. Monte Carlo

```- First-visit Monte Carlo with epsilon-greedy policy

Episodes to Stable Policy:- Bounded state discretization

├── Actor-Critic: ~600 (100% No Action strategy)- Epsilon decay for exploration-exploitation balance

├── Q-Learning:   ~800 (70% No Action, still exploring)

└── Monte Carlo:  >1000 (60% No Action, high exploration)### 2. Q-Learning

```- Off-policy TD learning

- Learning rate decay

### Variance-Performance Tradeoff- Epsilon-greedy exploration



| Metric | Indicates |### 3. Actor-Critic

|--------|-----------|- Policy gradient with value function baseline

| Low variance (±4) | Policy converged (deterministic) |- Entropy regularization for exploration

| High variance (±40) | Still exploring (stochastic) |- Normalized returns for stable learning



---## ✅ Fixes Applied



## 🎓 Educational Value### Critical Fixes

1. **Gymnasium API Compliance**

### What Students Learn   - ✅ `reset()` returns `(observation, info)`

   - ✅ `step()` returns `(obs, reward, terminated, truncated, info)`

1. **RL Fundamentals**   - ✅ Spaces are attributes, not properties

   - Value-based vs policy-based methods

   - Exploration-exploitation tradeoff2. **Reward Double-Counting Bug**

   - Convergence analysis   - ✅ Configuration cost no longer subtracted twice

   - ✅ Consistent reward calculation

2. **Deep RL**

   - Policy gradient methods3. **State Space Issues**

   - Actor-Critic architecture   - ✅ Bounded discretization (prevents infinite state space)

   - Neural network optimization   - ✅ Consistent 6D state across all files

   - ✅ Proper normalization for neural networks

3. **Practical Skills**

   - Gymnasium API usage4. **Agent Improvements**

   - PyTorch implementation   - ✅ Epsilon and learning rate decay

   - Hyperparameter tuning   - ✅ Better Actor-Critic architecture

   - Performance evaluation   - ✅ Entropy regularization

   - ✅ Gradient clipping

4. **Game Design**

   - Reward shaping5. **Info Dictionary**

   - Action space design   - ✅ Consistent structure across all code paths

   - Optimal strategy analysis   - ✅ Comprehensive metrics for debugging



---See `ISSUES_AND_FIXES.md` for detailed analysis.



## 🔮 Future Work## 📊 Expected Results



### Option 1: Accept Current Results ✅After training (500 episodes):

- **Monte Carlo**: ~45-55 score

**Status:** **RECOMMENDED** - Current implementation is complete- **Q-Learning**: ~50-60 score  

- **Actor-Critic**: ~55-70 score

- Document findings (✅ Done: [ANALYSIS_REPORT.md](./ANALYSIS_REPORT.md))

- Publish resultsResults may vary due to stochastic environment.

- Use as baseline for comparisons

- Educational demonstration## 🧪 Testing



### Option 2: Rebalance Game 🎮The `test_environment.py` file includes comprehensive tests:

- Gymnasium API compliance

**Status:** Proposed - See [REBALANCING_GUIDE.md](./REBALANCING_GUIDE.md)- Seeding reproducibility

- State validity

**Quick changes for action diversity:**- Action effects

```python- Info dictionary consistency

# train_game_env.py- Episode completion

initial_capacity = 10  # Force expansion (was 25)- No double-counting verification

capacity_match_bonus = 100.0 * rush_hour_multiplier  # Incentivize optimization (was 25.0)

penalty_missed = 10.0 * (missed_passengers ** 1.5)  # Punish shortfalls (was 3.0 × ^1.2)## 📝 Usage Example

```

```python

**Expected outcome:** 40-60% action usage, more strategic gameplayfrom train_game_env import TrainGameEnv



---# Create environment

env = TrainGameEnv(seed=42)

## 🧪 Experimental Reproducibility

# Reset

### Reproduction Stepsobs, info = env.reset()



```bash# Run episode

# 1. Clone repositorydone = False

git clone https://github.com/yourusername/RL-PROJECT.gitwhile not done:

cd RL-PROJECT    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

# 2. Setup environment    done = terminated or truncated

conda activate pt1

# Get final score

# 3. Train (uses fixed seed=42)score, raw_score = env.final_score()

cd scriptsprint(f"Score: {score}/100")

python train.py --agent all --episodes 1000```



# 4. Evaluate## 🔧 Customization

python evaluate.py --all --episodes 100

Adjust hyperparameters in environment:

# 5. Compare results with ANALYSIS_REPORT.md```python

```env = TrainGameEnv(

    initial_capacity=100,      # Starting capacity

### Key Parameters    seed=42,                   # Random seed

    verbose=False,             # Print debug info

- **Random Seed:** 42 (fixed for reproducibility)    render_mode='human'        # Enable rendering

- **Episodes:** 1000 training, 100 evaluation)

- **Initial Capacity:** 25```

- **Action Costs:** Add=2.0, Widen=1.0, NoAction=0.0

- **Hyperparameters:** See `scripts/config.py`## 📚 References



---- [Gymnasium Documentation](https://gymnasium.farama.org/)

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## 🛠️ Troubleshooting- Sutton & Barto - Reinforcement Learning: An Introduction



### Common Issues## 🤝 Contributing



**"ModuleNotFoundError: No module named 'gymnasium'"**This is a learning project. Feel free to experiment with:

```bash- Different reward structures

conda install gymnasium- New agent algorithms (PPO, DQN, A3C)

```- Environment variations

- Better visualization

**"Import torch could not be resolved"**

```bash## 📄 License

conda install pytorch

```Educational project - free to use and modify.



**"No trained agents found"**---

```bash

cd scripts**Last Updated**: October 11, 2025

python train.py --agent all --episodes 1000
```

**"KeyError: 'final_score'"**
- Fixed in current version
- Use `env.final_score()` method, not info dict

**TensorBoard not showing data**
```bash
# Check log directory
ls -la ../runs/

# Restart TensorBoard
tensorboard --logdir=../runs --reload_interval=5
```

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{train-capacity-rl-2025,
  author = {Your Name},
  title = {Train Capacity Management with Reinforcement Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/RL-PROJECT}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional RL algorithms (PPO, SAC, DQN)
- Real-world ridership data integration
- Multi-agent scenarios
- Transfer learning experiments
- Alternative reward structures

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👥 Credits

**Environment:** LRT-2 Manila Metro (13 stations)  
**Algorithms:** Monte Carlo, Q-Learning, Actor-Critic  
**Framework:** PyTorch + Gymnasium  
**Visualization:** Matplotlib + TensorBoard

---

## 📞 Contact

Questions? Open an issue or reach out!

---

**Status:** ✅ **Complete - Results Accepted**  
**Last Updated:** October 18, 2025  
**Version:** 1.0

*Happy training! 🚂*
