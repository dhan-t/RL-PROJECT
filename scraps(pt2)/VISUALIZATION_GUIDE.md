# Visualization Quick Start

## 🎨 Generate Beautiful Plots in 3 Commands

### Step 1: Train Agents (with visualization support)
```bash
python train.py --agent all --episodes 1000
```
This automatically saves training history for plotting!

### Step 2: Generate Comparison Plots
```bash
python visualize.py --mode comparison
```
**Output:** `../visualizations/agent_comparison.png`

Shows:
- 📊 Score comparison (mean ± std)
- 💰 Configuration costs
- 📈 Score distributions (violin plots)
- 🎯 Action usage breakdown

### Step 3: Generate Training Curves
```bash
python visualize.py --mode training
```
**Output:** `../visualizations/training_curves.png`

Shows:
- 📈 Learning curves (50-episode moving average)
- 💵 Cost progression over time
- 🎁 Reward accumulation
- 📦 Score distributions
- 🔄 Early vs late learning
- ⚡ Efficiency (score/cost)
- 📊 Performance consistency

---

## 🚀 One-Liner (Train + Visualize)

```bash
python train.py --agent all --episodes 1000 && python visualize.py --mode all
```

---

## 📖 View Interactive Guide

```bash
python viz_guide.py
```

---

## 💡 Advanced Options

### Show plots interactively (don't save files)
```bash
python visualize.py --mode comparison --show
```

### Custom output directory
```bash
python visualize.py --mode all --output-dir ./my_plots
```

### Use agents from custom directory
```bash
python visualize.py --save-dir ./my_models --mode all
```

---

## 📊 What You'll See

### Agent Comparison Plot
```
┌─────────────────────────────────────────────┐
│  Average Scores (with error bars)          │
│  ▓▓▓ Monte Carlo: 24.7 ± 37.7              │
│  ▓▓▓▓▓▓ Q-Learning: 72.6 ± 33.6            │
│  ▓▓▓▓▓▓▓ Actor-Critic: 70.2 ± 25.5         │
├─────────────────────────────────────────────┤
│  Action Distribution (stacked bars)         │
│  Shows which actions each agent prefers     │
└─────────────────────────────────────────────┘
```

### Training Curves Dashboard
```
┌────────────────────────────────────────────┐
│ 7 subplots showing comprehensive analysis: │
│ • Learning progress                        │
│ • Cost efficiency                          │
│ • Reward trends                            │
│ • Performance consistency                  │
└────────────────────────────────────────────┘
```

---

## 🔍 Interpreting Results

### Score Comparison
- **Higher = Better** performance
- **Lower variance** = more consistent
- Compare error bars to see stability

### Action Distribution
- **100% No Action** = conservative strategy
- **Balanced distribution** = adaptive strategy
- **100% Add Carriage** = aggressive expansion

### Learning Curves
- **Upward trend** = successful learning
- **Flat line** = converged or stuck
- **Oscillation** = high exploration

### Efficiency (Score/Cost)
- **Higher = Better** resource usage
- Shows which agent gets best results per dollar spent

---

## 🎯 Example Workflow

```bash
# 1. Clean old agents
rm ../saved_agents/*.pkl ../saved_agents/*.pt

# 2. Train fresh
python train.py --agent all --episodes 1000

# 3. Evaluate
python evaluate.py --all --episodes 100

# 4. Visualize everything
python visualize.py --mode all

# 5. Check the plots!
open ../visualizations/
```

---

## 📦 Requirements

Make sure you have matplotlib installed:
```bash
pip install matplotlib numpy
```

If you see import errors, install with:
```bash
conda install matplotlib numpy
```

---

## ❓ Troubleshooting

**"No training history found"**
- Solution: Train with `python train.py --agent all` first

**"No trained agents found"**
- Solution: Make sure agents exist in `../saved_agents/`
- Check with: `ls ../saved_agents/`

**Import errors (matplotlib)**
- Solution: `pip install matplotlib` or `conda install matplotlib`

**Plots look weird**
- Try: `python visualize.py --mode comparison --show` for interactive
- Adjust figure size in `visualize.py` if needed

---

## 🎨 Customization

Want to modify plots? Edit `visualize.py`:
- Line 30: Change colors
- Line 40: Adjust figure sizes
- Line 100: Modify smoothing window
- Line 200: Add new metrics

Happy visualizing! 📊✨
