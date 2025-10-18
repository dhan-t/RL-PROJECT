# 🔄 Game Rebalancing Guide
## Future Exploration: Designing for Action Diversity

**Purpose:** This guide provides recommendations for rebalancing the game to encourage strategic action variety while maintaining the core RL learning objectives.

**Status:** 📋 **PROPOSED** - Not yet implemented  
**Prerequisite:** Review [ANALYSIS_REPORT.md](./ANALYSIS_REPORT.md) for current results

---

## 🎯 Rebalancing Objectives

### Current Problem
Actor-Critic discovered that **100% No Action** is mathematically optimal:
- Initial capacity (25) handles most stations adequately
- Action costs (1.0-2.0) exceed expected benefits
- Missing passengers is cheaper than expanding capacity

### Desired Outcome
Create game mechanics that reward **strategic capacity management**:
- Active decision-making during rush hours
- Penalties for over/under capacity
- Rewards for perfect capacity matching
- Dynamic constraints that force adaptation

---

## 🔧 Proposed Changes

### Strategy 1: Force Early Decisions (Difficulty: ⭐☆☆☆☆)

**Change:** Lower initial capacity

```python
# Current (train_game_env.py, line 26)
def __init__(self, initial_capacity=25, ...):

# Proposed
def __init__(self, initial_capacity=10, ...):  # Was 25
```

**Effect:**
- Starting capacity (10) insufficient for regular stations (10-70 arrivals)
- Forces agents to expand capacity early
- Creates early-game strategic choices

**Expected Results:**
- Actor-Critic: 60-80% actions (capacity building phase)
- Q-Learning: 50-70% actions
- Monte Carlo: 70-90% actions

**Risk:** ⚠️ May still converge to "expand once then stop"

---

### Strategy 2: Increase Rush Hour Rewards (Difficulty: ⭐⭐☆☆☆)

**Change:** Make rush hour bonuses more valuable

```python
# Current (train_game_env.py, lines 323-330)
if 0.95 <= utilization <= 1.0:
    capacity_match_bonus = 25.0 * rush_hour_multiplier  # 25-50

# Proposed
if 0.95 <= utilization <= 1.0:
    capacity_match_bonus = 100.0 * rush_hour_multiplier  # 100-200!
```

**Effect:**
- Perfect capacity matching during rush hour = huge reward
- Incentivizes precise capacity management
- Makes optimization worthwhile

**Expected Results:**
- Actor-Critic: 30-50% actions (optimizes for rush hours)
- Agents learn time-of-day strategies

**Risk:** ⚠️ May create "yo-yo" strategy (expand for rush, shrink after)

---

### Strategy 3: Exponential Missed Passenger Penalties (Difficulty: ⭐⭐☆☆☆)

**Change:** Punish capacity shortfalls heavily

```python
# Current (train_game_env.py, line 302)
penalty_missed = 3.0 * (missed_passengers ** 1.2)

# Proposed
penalty_missed = 10.0 * (missed_passengers ** 1.5)  # Much steeper!
```

**Effect:**
- Missing 50 passengers becomes extremely costly
- Forces agents to maintain adequate capacity
- Prevents "ignore passengers" strategy

**Expected Results:**
- Actor-Critic: 40-60% actions (maintains capacity)
- Reduces "do nothing" viability

**Risk:** ⚠️ May lead to over-expansion (waste capacity)

---

### Strategy 4: Aggressive Capacity Decay (Difficulty: ⭐⭐⭐☆☆)

**Change:** Capacity deteriorates faster

```python
# Current (train_game_env.py, lines 287-289)
if self.steps > 0 and self.steps % 100 == 0:
    self.capacity = max(0, self.capacity - 2)

# Proposed
if self.steps > 0 and self.steps % 30 == 0:  # Every 30 steps
    self.capacity = max(5, self.capacity - 5)  # Lose 5 capacity
```

**Effect:**
- Capacity constantly deteriorating
- Requires ongoing maintenance actions
- "No Action" becomes unsustainable

**Expected Results:**
- All agents: 50-80% actions (constant maintenance)
- Introduces "maintenance tax" concept

**Risk:** ⚠️ May be frustrating if too aggressive

---

### Strategy 5: Dynamic Demand Surges (Difficulty: ⭐⭐⭐⭐☆)

**Change:** Unpredictable passenger spikes

```python
# New function in train_game_env.py
def _simulate_arrivals(self):
    amin, amax = self._arrival_bounds(self.station_idx)
    base = random.randint(amin, amax)
    
    current_hour = self.sim_minutes // 60
    mult = self._time_multiplier(current_hour)
    
    # NEW: Random surge events
    if random.random() < 0.05:  # 5% chance
        mult *= 3.0  # Triple passengers!
        print(f"🚨 SURGE EVENT at {self.stations[self.station_idx]}")
    
    surge_factor = 1.0 + (self.steps / 2000) * 2.0
    return max(0, int(round(base * mult * surge_factor)))
```

**Effect:**
- Unpredictable demand spikes
- Fixed capacity strategies fail
- Requires adaptive decision-making

**Expected Results:**
- Actor-Critic: Must learn probabilistic policies
- Encourages diverse action distributions
- Tests generalization

**Risk:** ⚠️⚠️ Increases variance, harder to learn

---

### Strategy 6: Multi-Objective Rewards (Difficulty: ⭐⭐⭐⭐⭐)

**Change:** Balance multiple competing goals

```python
# New reward structure
step_reward = (
    1.5 * boarded                          # Boarding reward
    - penalty_unused                        # Waste penalty
    - config_penalty                        # Cost penalty
    - penalty_missed                        # Missed penalty
    + capacity_match_bonus                  # Efficiency bonus
    + passenger_satisfaction_bonus          # NEW: Happiness metric
    - environmental_cost                    # NEW: Carbon footprint
)

# Where:
passenger_satisfaction = (boarded / arrivals) * 50  # Scale 0-50
environmental_cost = (capacity / 100) * 0.5  # Bigger trains = pollution
```

**Effect:**
- Must balance efficiency, satisfaction, environment
- No single metric dominates
- Rich strategic landscape

**Expected Results:**
- Complex, nuanced strategies
- Different agents find different tradeoffs
- Educational value maximized

**Risk:** ⚠️⚠️⚠️ Much harder to tune, requires extensive testing

---

## 🧪 Recommended Rebalancing Path

### Phase 1: Quick Wins (1-2 hours)

**Implement:**
1. Strategy 1 (Lower initial capacity to 10)
2. Strategy 2 (Increase rush hour bonuses to 100.0)

**Test:**
```bash
# Backup current environment
cp train_game_env.py train_game_env_backup.py

# Make changes
# Edit train_game_env.py lines 26, 325

# Retrain
rm saved_agents/*.pkl saved_agents/*.pt
python train_with_tensorboard.py --agent all --episodes 1000

# Compare
python visualize.py --mode all
python evaluate.py --all
```

**Expected Outcome:**
- Actor-Critic: 30-50% actions
- Q-Learning: 40-60% actions
- Clearer strategic differences

---

### Phase 2: Medium Changes (2-4 hours)

**Add to Phase 1:**
3. Strategy 3 (Exponential penalties: 10.0 × missed^1.5)
4. Strategy 4 (Decay every 30 steps, -5 capacity)

**Test:**
```bash
# Retrain with new parameters
python train_with_tensorboard.py --agent all --episodes 1500

# Longer training needed for adaptation
```

**Expected Outcome:**
- All agents: 50-70% actions
- More dynamic gameplay
- Higher variance initially

---

### Phase 3: Advanced (4-8 hours)

**Add to Phase 2:**
5. Strategy 5 (Surge events)
6. Strategy 6 (Multi-objective rewards)

**Test:**
```bash
# Extended training for complex strategies
python train_with_tensorboard.py --agent all --episodes 2000
```

**Expected Outcome:**
- Rich strategic diversity
- Publication-worthy results
- Significant learning insights

---

## 📊 Success Metrics

### Before Rebalancing (Current)
- Actor-Critic: 100% No Action, 56.9 score
- Action diversity: 0%
- Strategic depth: Low

### After Rebalancing (Target)
- Actor-Critic: 40-60% actions
- Action diversity: 40-60%
- Strategic depth: High
- Score variance: Moderate (±15-25)
- All agents show distinct strategies

---

## 🔬 Experimental Design

### Hypothesis
> "Rebalanced game mechanics will increase action diversity while maintaining clear performance differences between algorithms."

### Variables

**Independent:**
- Initial capacity
- Rush hour bonuses
- Penalty structure
- Decay rate

**Dependent:**
- Action distribution (%)
- Average score
- Score variance
- Convergence speed

**Control:**
- Random seed (42)
- Number of episodes (1000)
- Agent architectures
- Evaluation protocol (100 episodes)

### Analysis Plan

1. **Baseline:** Document current performance
2. **Implement:** Apply rebalancing changes
3. **Train:** Retrain all agents
4. **Evaluate:** 100-episode evaluation
5. **Compare:** Before/after analysis
6. **Visualize:** Generate comparison plots
7. **Document:** Update findings

---

## 📈 Expected Learning Outcomes

### For Students

**Before Rebalancing:**
- ✅ Understand convergence
- ✅ Compare algorithms
- ✅ Analyze optimal policies
- ⚠️ Limited strategic variety

**After Rebalancing:**
- ✅ Understand convergence
- ✅ Compare algorithms
- ✅ Analyze optimal policies
- ✅ **Explore complex strategies**
- ✅ **See time-dependent policies**
- ✅ **Observe risk-reward tradeoffs**

---

## 🎯 Implementation Checklist

### Pre-Implementation
- [ ] Review current results ([ANALYSIS_REPORT.md](./ANALYSIS_REPORT.md))
- [ ] Backup original environment (`cp train_game_env.py train_game_env_backup.py`)
- [ ] Choose rebalancing strategy (recommend Phase 1)
- [ ] Set up TensorBoard logging

### Implementation
- [ ] Modify `train_game_env.py` parameters
- [ ] Update documentation with new parameters
- [ ] Test environment with random agent (smoke test)
- [ ] Verify game is still winnable

### Training
- [ ] Clear old saved agents
- [ ] Train with TensorBoard: `python train_with_tensorboard.py --agent all --episodes 1000`
- [ ] Monitor learning curves in real-time
- [ ] Save checkpoints at key milestones

### Evaluation
- [ ] Evaluate all agents: `python evaluate.py --all --episodes 100`
- [ ] Generate visualizations: `python visualize.py --mode all`
- [ ] Compare with baseline results
- [ ] Document findings

### Analysis
- [ ] Compare action distributions
- [ ] Analyze strategy changes
- [ ] Measure convergence differences
- [ ] Update documentation

---

## 💡 Tips for Successful Rebalancing

### Do's ✅
- **Start small:** Implement one change at a time
- **Baseline first:** Always compare against original results
- **Use TensorBoard:** Monitor learning in real-time
- **Save checkpoints:** Keep model snapshots at different episodes
- **Document everything:** Note all parameter changes
- **Test incrementally:** Verify each change before adding more

### Don'ts ❌
- **Don't change everything at once:** Hard to debug
- **Don't skip evaluation:** Always run 100-episode eval
- **Don't ignore variance:** High variance = still learning
- **Don't overtune:** Some randomness is educational
- **Don't lose baseline:** Keep original results for comparison

---

## 🔍 Debugging Guide

### Problem: Agents still converge to 100% No Action

**Solutions:**
1. Lower initial capacity more (try 5)
2. Increase rush hour bonuses more (try 200.0)
3. Add mandatory capacity requirements
4. Increase episode length (more rush hours)

### Problem: All agents perform poorly

**Solutions:**
1. Rewards too punishing → reduce penalty multipliers
2. Game too hard → increase initial capacity slightly
3. Not enough training → try 2000 episodes
4. Learning rate issues → check hyperparameters

### Problem: High variance doesn't decrease

**Solutions:**
1. Game is stochastic → this is normal!
2. Surge events too frequent → reduce probability
3. Penalties inconsistent → smooth reward function
4. Need more episodes → try 1500-2000

---

## 📚 Further Reading

### Research Topics
1. **Multi-objective RL:** Balancing competing rewards
2. **Non-stationary environments:** Adapting to changing dynamics
3. **Risk-sensitive RL:** Conservative vs aggressive strategies
4. **Transfer learning:** Applying learned policies to new scenarios

### Related Work
- OpenAI Gym classic control environments
- DeepMind's Atari benchmark
- Real-world capacity planning problems
- Dynamic pricing in ride-sharing

---

## 🎓 Educational Extensions

### For Advanced Students

1. **Curriculum Learning:**
   - Start with easy game (high initial capacity)
   - Gradually make harder (lower capacity, higher penalties)
   - Train agent across difficulty progression

2. **Meta-Learning:**
   - Train agents on multiple game configurations
   - Learn to adapt quickly to new parameters
   - Measure transfer learning efficiency

3. **Multi-Agent:**
   - Multiple trains competing for passengers
   - Cooperative vs competitive scenarios
   - Nash equilibrium analysis

4. **Real-World Data:**
   - Use actual LRT-2 ridership patterns
   - Time-series forecasting of demand
   - Integration with Manila traffic data

---

## 📌 Status

**Phase:** 📋 **PROPOSED**  
**Priority:** Optional enhancement  
**Effort:** 4-20 hours depending on complexity  
**Risk:** Low (original results preserved)

---

## 🚀 Quick Start

Ready to rebalance? Start here:

```bash
# 1. Backup
cp train_game_env.py train_game_env_backup.py

# 2. Edit train_game_env.py
# Line 26: initial_capacity = 10  (was 25)
# Line 325: capacity_match_bonus = 100.0 * rush_hour_multiplier  (was 25.0)

# 3. Train
rm saved_agents/*.pkl saved_agents/*.pt
python train_with_tensorboard.py --agent all --episodes 1000

# 4. Start TensorBoard (in another terminal)
tensorboard --logdir=../runs

# 5. Evaluate
python evaluate.py --all --episodes 100

# 6. Visualize
python visualize.py --mode all

# 7. Compare results!
```

---

*Guide created: October 18, 2025*  
*Last updated: October 18, 2025*  
*Version: 1.0*
