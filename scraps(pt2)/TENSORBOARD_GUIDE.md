# 🔥 TensorBoard Quick Start

## Launch TensorBoard in 2 Commands

### Step 1: Train with TensorBoard Logging

```bash
cd scripts
python train_with_tensorboard.py --agent all --episodes 1000
```

### Step 2: Start TensorBoard (in another terminal)

```bash
cd RL-PROJECT
tensorboard --logdir=runs
```

Then open: **http://localhost:6006**

---

## 📊 What You'll See

### SCALARS Tab

**Monte Carlo Metrics:**
- `MonteCarlo/Score` - Episode scores
- `MonteCarlo/Score_MA50` - 50-episode moving average
- `MonteCarlo/Score_Std` - Score standard deviation
- `MonteCarlo/ConfigCost` - Configuration spending
- `MonteCarlo/EpisodeReward` - Total episode reward
- `MonteCarlo/Epsilon` - Exploration rate

**Q-Learning Metrics:**
- `QLearning/Score`
- `QLearning/Score_MA50`
- `QLearning/Score_Std`
- `QLearning/ConfigCost`
- `QLearning/EpisodeReward`
- `QLearning/Epsilon` - Exploration rate
- `QLearning/Alpha` - Learning rate

**Actor-Critic Metrics:**
- `ActorCritic/Score`
- `ActorCritic/Score_MA50`
- `ActorCritic/Score_Std`
- `ActorCritic/ConfigCost`
- `ActorCritic/EpisodeReward`
- `ActorCritic/ActorLoss` - Policy gradient loss
- `ActorCritic/CriticLoss` - Value function loss

**Comparison:**
- `Comparison/FinalScore_*` - Final performance comparison
- `Comparison/FinalCost_*` - Cost comparison
- `Comparison/Consistency_*` - Variance comparison

---

## 🎯 How to Use TensorBoard

### Compare Learning Curves

1. Go to **SCALARS** tab
2. Search for: `Score_MA50`
3. You'll see 3 lines:
   - 🔴 Monte Carlo (red)
   - 🔵 Q-Learning (blue)
   - 🟢 Actor-Critic (green)

**What to Look For:**
- **Upward trend** = successful learning
- **Flat line** = converged or stuck
- **Spikes** = high variance / exploration

### Monitor Convergence

Search for: `Score_Std`

**Interpretation:**
- **Decreasing std** = agent converging
- **High std (>40)** = still exploring
- **Low std (<10)** = converged policy

### Check Exploration

Search for: `Epsilon`

**Expected Pattern:**
- Starts at 1.0 (100% random)
- Decays to 0.01 (1% exploration)
- Actor-Critic doesn't use epsilon

### View Loss Functions

Search for: `Loss`

**Actor-Critic only:**
- `ActorLoss` should decrease (policy improving)
- `CriticLoss` should stabilize (value function learned)

---

## ⚙️ Advanced Usage

### Compare Multiple Runs

```bash
# Run 1: Default parameters
python train_with_tensorboard.py --agent all --episodes 1000

# Run 2: Different seed
python train_with_tensorboard.py --agent all --episodes 1000 --seed 123

# Run 3: More episodes
python train_with_tensorboard.py --agent all --episodes 2000

# View all runs in TensorBoard
tensorboard --logdir=../runs
```

TensorBoard automatically detects multiple runs!

### Smooth Curves

In TensorBoard UI:
- Adjust **Smoothing** slider (bottom left)
- 0.0 = raw data
- 0.9 = very smooth

### Download Data

Click "Show data download links" (top right) to export CSV

---

## 🎨 TensorBoard Tips

### Organize Your Runs

```bash
# Name runs descriptively
python train_with_tensorboard.py --agent all --episodes 1000 --log-dir ../runs/baseline_1000ep

python train_with_tensorboard.py --agent all --episodes 1000 --log-dir ../runs/rebalanced_1000ep
```

### Compare Across Time

TensorBoard shows timestamps - hover over runs to see training date/time

### Filter Metrics

Use the search box to filter:
- `Score` - Show only score metrics
- `ActorCritic/` - Show only Actor-Critic
- `MA50` - Show only moving averages

---

## 📈 Example Analysis

### Question: "Which agent learns fastest?"

**Steps:**
1. Go to SCALARS tab
2. Search: `Score_MA50`
3. Look for first agent to reach plateau

**Expected Answer:**
- **Actor-Critic**: Reaches ~55 by episode 600
- **Q-Learning**: Reaches ~55 by episode 800
- **Monte Carlo**: Still climbing at episode 1000

### Question: "Which agent is most consistent?"

**Steps:**
1. Search: `Score_Std`
2. Check final values (episode 1000)

**Expected Answer:**
- **Actor-Critic**: ~4.0 (most consistent)
- **Q-Learning**: ~42.0 
- **Monte Carlo**: ~43.0

### Question: "Is Actor-Critic still learning?"

**Steps:**
1. Search: `ActorLoss` and `CriticLoss`
2. Check if losses are decreasing

**Expected Answer:**
- Both losses plateau after ~600 episodes
- Indicates convergence

---

## 🔧 Troubleshooting

**TensorBoard shows "No dashboards are active"**
- Wait a few seconds for data to load
- Check log directory: `ls ../runs/`
- Make sure training has started

**TensorBoard port already in use**
```bash
tensorboard --logdir=../runs --port=6007
```

**Old runs cluttering view**
```bash
# Clear old runs
rm -rf ../runs/*

# Or organize by date
mkdir ../runs/archive_2025-10-18
mv ../runs/train_* ../runs/archive_2025-10-18/
```

**Curves look noisy**
- Increase smoothing slider
- View moving average metrics (`_MA50`)

---

## 🎓 Learning Exercise

### Compare Rebalanced vs Original

1. **Train baseline:**
```bash
python train_with_tensorboard.py --agent all --episodes 1000 --log-dir ../runs/baseline
```

2. **Modify environment** (see REBALANCING_GUIDE.md)

3. **Train rebalanced:**
```bash
python train_with_tensorboard.py --agent all --episodes 1000 --log-dir ../runs/rebalanced
```

4. **Compare in TensorBoard:**
```bash
tensorboard --logdir=../runs
```

Look for:
- Score differences
- Action diversity (via cost patterns)
- Convergence speed changes

---

## 📸 Taking Screenshots

1. Arrange plots as desired in TensorBoard
2. Click camera icon (top right of each plot)
3. Or use OS screenshot tool

---

## 🚀 Next Steps

1. ✅ Train agents with TensorBoard
2. ✅ Monitor learning in real-time
3. ✅ Compare algorithms visually
4. ✅ Download data for further analysis
5. ✅ Experiment with hyperparameters
6. ✅ Document findings

---

**Pro Tip:** Keep TensorBoard running while training to watch learning happen in real-time! 📈

---

*For more details, see: [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)*
