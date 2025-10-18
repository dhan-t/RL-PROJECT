# 🎮 Train Game Balance Guide

## 🏆 Training Results: Q-Learning is the Champion!

### Final Scores (1000 Episodes)

| Agent | Avg Score | Std Dev | Config Cost | Total Reward | Rank |
|-------|-----------|---------|-------------|--------------|------|
| **Q-Learning** | **93.6** | **±3.2** ✅ | 45.0 | 10,895.8 | 🥇 **1st** |
| Actor-Critic | 84.3 | ±10.2 | 55.0 | 6,306.8 | 🥈 2nd |
| Monte Carlo | 33.5 | ±39.9 | 331.0 | -63,910.3 | 🥉 3rd |

### Why Q-Learning Won:
1. ✅ **Highest average score** (93.6 vs 84.3)
2. ✅ **Most consistent** (only ±3.2 variance)
3. ✅ **Best total reward** (10,895 vs 6,306)
4. ✅ **Lowest config cost** (45 vs 55)

### Why Monte Carlo Failed:
- ❌ Extreme variance (±39.9) - very unstable
- ❌ Negative total reward (-63,910)
- ❌ Wasted money on unnecessary configs (331 cost)
- ❌ Poor state discretization for this sequential problem

---

## 🎯 Current Game Behavior: "No Action" Dominance

### Why Agents Choose "No Action" ~90% of Time

The current reward function:
```python
reward = (passengers_boarded × 1.5) - (unused_capacity × 0.5) - config_cost
```

**Economic Analysis:**

| Action | Cost | Capacity Added | Break-Even Point |
|--------|------|----------------|------------------|
| Add Carriage | -10 | +100 | Need 67+ passengers |
| Widen Carriage | -5 | +50 | Need 34+ passengers |
| No Action | 0 | 0 | Always safe ✅ |

**Result:** Starting with 100 capacity and never changing is economically optimal!

This is actually **smart strategy**, not a bug. But it makes the game less interesting.

---

## 💡 Solutions: How to Encourage Action Variety

### Option A: Reduce Configuration Costs (Easiest) ⭐

**File:** `train_game_env.py` (around line 238-248)

```python
# BEFORE:
if action == 0:  # Add Carriage
    cost, weight = 10.0, 100.0
elif action == 1:  # Widen Carriage
    cost, weight = 5.0, 50.0

# AFTER:
if action == 0:  # Add Carriage
    cost, weight = 3.0, 100.0  # ✅ 70% cheaper
elif action == 1:  # Widen Carriage
    cost, weight = 1.5, 50.0   # ✅ 70% cheaper
```

**Expected Effect:**
- Add now needs only 20 passengers to break even
- Widen needs only 10 passengers to break even
- Action usage: Add ~20%, Widen ~15%, No Action ~65%

---

### Option B: Increase Passenger Demand (Moderate)

**File:** `train_game_env.py` (around line 150-180)

```python
# BEFORE:
if is_rush_hour:
    passengers_waiting = min(int(np.random.normal(100, 30)), 150)
else:
    passengers_waiting = min(int(np.random.normal(40, 15)), 80)

# AFTER:
if is_rush_hour:
    passengers_waiting = min(int(np.random.normal(200, 50)), 300)  # ✅ 2x demand
else:
    passengers_waiting = min(int(np.random.normal(60, 20)), 100)
```

**Expected Effect:**
- 100 capacity insufficient during rush hours
- Forces capacity upgrades
- More strategic timing of Add/Widen actions

---

### Option C: Add Management Bonus (Advanced)

**File:** `train_game_env.py` (in step function, around line 250)

```python
# Add this after determining cost
if action in [0, 1]:  # Add or Widen
    management_bonus = 5.0
else:  # No Action
    management_bonus = 0.0

# Update reward calculation
reward = (passengers_boarded * 1.5) - (unused_capacity * 0.5) - cost + management_bonus
```

**Expected Effect:**
- Effective cost: Add=5, Widen=0 (break-even at 33 and 0 passengers)
- Rewards proactive management
- Action usage: Add ~25%, Widen ~20%, No Action ~55%

---

### Option D: Station-Specific Surge Pricing (Realistic)

**File:** `train_game_env.py` (in step function, around line 250)

```python
# Define premium stations
premium_stations = [3, 6, 7, 11]  # J. Ruiz, Betty Go, Cubao, Marikina
passenger_value = 2.0 if self.current_station in premium_stations else 1.5

# Update reward
reward = (passengers_boarded * passenger_value) - (unused_capacity * 0.5) - cost
```

**Expected Effect:**
- Strategic capacity additions before premium stations
- Location-based decision making
- More complex agent strategies

---

### Option E: Capacity Decay (Game-Changer)

**File:** `train_game_env.py` (in step function, after action handling)

```python
# Add capacity decay every 10 stations
if self.station_visits % 10 == 0:
    decay = 20
    self.capacity = max(50, self.capacity - decay)
    info['capacity_decayed'] = decay
```

**Expected Effect:**
- Forces periodic capacity restoration
- Can't rely on "No Action" forever
- More dynamic gameplay

---

## 🎯 Recommended Configuration (Balanced)

Combine these changes for best results:

1. ✅ **Reduce costs**: Add=3.0, Widen=1.5
2. ✅ **Increase rush hour demand**: 200-300 passengers
3. ✅ **Add management bonus**: +5.0 for taking action
4. ✅ **Optional**: Premium stations pay 2.0x

**Expected Outcome:**
- Action distribution: No Action ~60%, Add ~20%, Widen ~15%
- Higher scores (90-98/100)
- More strategic gameplay
- Better agent learning curves

---

## 📋 Implementation Steps

### 1. Backup Current Code
```bash
cp train_game_env.py train_game_env_backup.py
```

### 2. Make Changes
Edit `train_game_env.py` with your chosen options above.

### 3. Clean Old Models
```bash
rm saved_agents/*.pt saved_agents/*.pkl
```

### 4. Retrain Agents
In notebook:
1. Restart kernel (important!)
2. Re-run Cell 2 (import environment)
3. Re-run Cell 5 (training)
4. Compare results!

### 5. Compare Performance
```python
# After retraining, check action distribution
# You should see more variety!
```

---

## 📊 Expected Results After Changes

### Before (Current):
- No Action: ~90% 😴
- Add Carriage: ~5%
- Widen Carriage: ~5%
- Average Score: 85-95/100

### After (Recommended Config):
- No Action: ~60% ✅
- Add Carriage: ~20% ✅
- Widen Carriage: ~15% ✅
- Average Score: 90-98/100 ⬆️

---

## 🔍 Why This Matters

The current "No Action dominance" isn't a bug—it's the **optimal solution** to the problem as defined. But it makes the RL system less interesting because:

1. ❌ Agents learn a single trivial strategy
2. ❌ No adaptive behavior to changing conditions
3. ❌ Doesn't demonstrate complex RL capabilities
4. ❌ Not realistic (real trains do adjust capacity)

By adjusting the game balance, you create a more interesting optimization problem that better showcases RL capabilities!

---

## 🚀 Quick Start: 3-Minute Test

**Simplest change to try:**

1. Open `train_game_env.py`
2. Find line ~240: `cost, weight = 10.0, 100.0`
3. Change to: `cost, weight = 3.0, 100.0`
4. Find line ~242: `cost, weight = 5.0, 50.0`
5. Change to: `cost, weight = 1.5, 50.0`
6. Save file
7. Restart notebook kernel
8. Re-run training (Cell 5)
9. Compare action distribution!

You should immediately see more Add/Widen actions! 🎉

---

## 📚 Further Reading

- Check `rl_training.ipynb` cells at the end for detailed analysis
- See `ACTION_SPACE_FIX.md` for the original 3-action system fix
- See `CRITICAL_FIX_NEEDED.md` for Cell 5 parameter bug explanation

---

**Remember:** The game isn't broken—it's just optimally solved! Now make it more challenging! 🚂
