# train_game_env.py
import random
import numpy as np

class TrainGameEnv:
    def __init__(self, initial_capacity=100, seed=None, verbose=False):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Core game state
        self.initial_capacity = initial_capacity
        self.capacity = initial_capacity
        self.passengers_onboard = 0

        # Performance tracking
        self.raw_score = 0.0
        self.total_boarded = 0
        self.total_unused = 0.0
        self.total_config_cost = 0.0
        self.peak_inefficiency = 0  # Track worst excess capacity

        # Station configuration (LRT-2 line)
        self.stations = [
            "Recto", "Legarda", "Pureza", "V. Mapa", "J. Ruiz", "Gilmore",
            "Betty Go", "Cubao", "Anonas", "Katipunan",
            "Santolan", "Marikina", "Antipolo"
        ]
        self.num_stations = len(self.stations)
        self.station_idx = 0
        self.direction = +1

        # Infrastructure stress management
        self.weight_window = []
        self.window_size = 10
        self.base_collapse_threshold = 10.0

        # Time management (04:00-22:00 operating hours)
        self.sim_minutes = random.randint(4 * 60, 22 * 60 - 1)
        self.steps = 0
        self.max_steps = 2000
        self.station_visits = 0

        # Efficiency tracking
        self.previous_onboard = 0

        self.done = False
        self.done_reason = None
        self.verbose = verbose

    # ----------------- core simulation methods -----------------
    def _time_multiplier(self, hour):
        """Get passenger multiplier based on time of day"""
        if 6 <= hour <= 8:   return 1.9  # Morning rush
        if 11 <= hour <= 13: return 1.6  # Lunch rush  
        if 17 <= hour <= 19: return 1.9  # Evening rush
        return 0.8 if random.random() < 0.45 else 1.0

    def _arrival_bounds(self, idx):
        """Get min/max arrivals for station type"""
        if idx in (0, self.num_stations-1, 7):  # Terminals + major hub
            return (40, 150)
        return (10, 70)

    def _simulate_arrivals(self):
        """Calculate arriving passengers at current station"""
        amin, amax = self._arrival_bounds(self.station_idx)
        base = random.randint(amin, amax)
        
        current_hour = self.sim_minutes // 60
        mult = self._time_multiplier(current_hour)

        # Passenger volume increases over time
        surge_factor = 1.0 + (self.steps / 2000) * 2.0
        return max(0, int(round(base * mult * surge_factor)))

    def _calculate_efficiency_penalty(self, unused_space, alighted_passengers, previous_onboard, current_hour):
        """Calculate penalty based on operational efficiency"""
        base_penalty = 0.3 * unused_space
        penalty_growth = 1.0 + (self.steps / 1000)
        
        # Reduced penalty for empty trains starting service
        if previous_onboard == 0:
            return base_penalty * penalty_growth * 0.1
        
        alighting_ratio = alighted_passengers / previous_onboard
        
        # Time-based efficiency expectations
        if 6 <= current_hour <= 8 or 17 <= current_hour <= 19:
            expected_efficiency = 0.7  # Rush hour
        elif 11 <= current_hour <= 13:
            expected_efficiency = 0.6  # Lunch hour
        else:
            expected_efficiency = 0.4  # Off-peak

        # Station-type efficiency expectations
        if self.station_idx in (0, self.num_stations-1):
            station_factor = 0.9  # Terminals
        elif self.station_idx == 7:
            station_factor = 0.8  # Major hub
        else:
            station_factor = 0.6  # Regular stations

        target_efficiency = max(0.3, min(0.95, expected_efficiency * station_factor))
        
        # Larger capacity = higher waste penalty
        capacity_factor = 0.5 + (self.capacity / 2000)
        
        # Efficiency-based penalty scaling
        if alighting_ratio > target_efficiency + 0.15:
            efficiency_multiplier = 0.1  # High efficiency
        elif alighting_ratio > target_efficiency:
            efficiency_multiplier = 0.3  # Good efficiency
        elif alighting_ratio > target_efficiency * 0.7:
            efficiency_multiplier = 0.6  # Moderate efficiency
        else:
            efficiency_multiplier = 1.2  # Poor efficiency
        
        efficiency_multiplier *= capacity_factor
        return base_penalty * penalty_growth * efficiency_multiplier

    def reset(self):
        """Reset environment to initial state"""
        self.capacity = self.initial_capacity
        self.passengers_onboard = 0
        self.raw_score = 0.0
        self.total_boarded = 0
        self.total_unused = 0.0
        self.total_config_cost = 0.0
        self.peak_inefficiency = 0
        self.station_idx = 0
        self.direction = +1
        self.weight_window = []
        
        self.sim_minutes = random.randint(4 * 60, 22 * 60 - 1)
        self.steps = 0
        self.station_visits = 0
        self.previous_onboard = 0
        self.done = False
        self.done_reason = None
        return self._get_state()

    def _get_state(self):
        """Get current environment state vector"""
        hour = self.sim_minutes // 60
        minute = self.sim_minutes % 60
        return np.array([
            float(self.capacity),
            float(self.passengers_onboard),
            float(self.station_idx),
            float(self.direction),
            float(hour),
            float(minute)
        ], dtype=np.float32)

    def step(self, action):
        """Execute one game step with given action"""
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        alighted_passengers = 0
        self.previous_onboard = self.passengers_onboard

        # Process capacity actions
        if action == 0:  # Add carriage (+100 capacity)
            self.capacity += 100
            cost, weight = 10.0, 1.0
        elif action == 1:  # Widen carriage (+50 capacity)  
            self.capacity += 50
            cost, weight = 5.0, 0.5
        else:  # No action
            cost, weight = 0.0, 0.0

        # Apply configuration cost penalty
        config_penalty = 2.0 * cost
        self.total_config_cost += cost
        self.raw_score -= config_penalty

        # Check for infrastructure collapse
        collapse_threshold = max(3.0, self.base_collapse_threshold - (self.steps / 200))
        self.weight_window.append(weight)
        if len(self.weight_window) > self.window_size:
            self.weight_window.pop(0)
        if sum(self.weight_window) >= collapse_threshold:
            self.done = True
            self.done_reason = f"Collapse at station {self.stations[self.station_idx]}"
            self.raw_score -= 500.0
            return self._get_state(), -1000.0, True, {"reason": self.done_reason, "alighted": 0}

        # Passenger alighting simulation
        if self.passengers_onboard > 0:
            alighted_passengers = random.randint(0, self.passengers_onboard)
            self.passengers_onboard -= alighted_passengers

        # Clear train at terminal stations
        if self.station_idx in (0, self.num_stations-1):
            self.passengers_onboard = 0

        # Passenger boarding simulation
        arrivals = self._simulate_arrivals()
        space = max(0, self.capacity - self.passengers_onboard)
        boarded = min(arrivals, space)
        self.passengers_onboard += boarded

        unused = max(0, self.capacity - self.passengers_onboard)
        self.peak_inefficiency = max(self.peak_inefficiency, unused)

        # Calculate rewards and penalties
        current_hour = self.sim_minutes // 60
        penalty_unused = self._calculate_efficiency_penalty(
            unused, alighted_passengers, self.previous_onboard, current_hour
        )
        
        reward_board = 1.5 * boarded
        station_reward = reward_board - penalty_unused

        # Update game state
        self.raw_score += station_reward
        self.total_boarded += boarded
        self.total_unused += unused
        self.station_visits += 1
        self.steps += 1

        # Advance simulation time
        self.sim_minutes += 5
        if self.sim_minutes >= 22 * 60:
            self.done = True
            self.done_reason = "End of operating hours (22:00)"

        # Move to next station
        next_idx = self.station_idx + self.direction
        if next_idx < 0 or next_idx >= self.num_stations:
            self.direction *= -1
            next_idx = self.station_idx + self.direction
            self.passengers_onboard = 0
        self.station_idx = next_idx

        # Check step limit
        if self.steps >= self.max_steps:
            self.done = True
            self.done_reason = "Max steps reached."

        return self._get_state(), station_reward - (0.2 * cost), self.done, {
            "alighted": alighted_passengers,
            "penalty_unused": penalty_unused,
            "efficiency_ratio": alighted_passengers / max(1, self.previous_onboard)
        }

    def final_score(self):
        """Calculate final normalized score (1-100)"""
        distance_bonus = self.station_visits * 10
        efficiency_penalty = self.peak_inefficiency * 0.1
        config_efficiency = max(0, 1 - (self.total_config_cost / (self.total_boarded + 1)))
        
        effective_score = (self.raw_score + distance_bonus - efficiency_penalty) * config_efficiency
        
        # Normalize to 1-100 scale
        S_min = -80 * self.station_visits
        S_max = 100 * self.station_visits
        normalized = round(1 + ((effective_score - S_min) / (S_max - S_min)) * 99)
        
        return max(1, min(100, normalized)), effective_score

    # RL-SPECIFIC METHODS
    @property
    def observation_space(self):
        """Define the observation space for RL"""
        from gymnasium import spaces
        return spaces.Box(
            low=np.array([0, 0, 0, -1, 0, 0]),  # min values for each state element
            high=np.array([np.inf, np.inf, 12, 1, 23, 59]),  # max values
            dtype=np.float32
        )
    
    @property 
    def action_space(self):
        """Define the action space for RL"""
        from gymnasium import spaces
        return spaces.Discrete(3)  # 0, 1, 2
    
    def get_info(self):
        """Get additional info for monitoring"""
        return {
            'total_boarded': self.total_boarded,
            'total_config_cost': self.total_config_cost,
            'average_efficiency': self.total_boarded / max(1, self.capacity * self.steps),
            'peak_inefficiency': self.peak_inefficiency
        }

    def render(self, mode='human'):
        """Render the environment for visualization"""
        return draw_train(self)

    def close(self):
        """Clean up resources"""
        pass


# ===============================
# VISUALIZATION (OUTSIDE THE CLASS)
# ===============================
def draw_train(env):
    """Display current train position and status"""
    track = ["-"] * env.num_stations
    idx = env.station_idx
    track[idx] = "🚉"
    train = "🚂" if env.direction == 1 else "🚋"
    print("\n" + "".join(track))
    print(" " * idx + train)
    print(f"📍 {env.stations[idx]} | Cap: {env.capacity} | Onboard: {env.passengers_onboard}")