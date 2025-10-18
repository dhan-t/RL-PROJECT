# train_game_env.py
# Gymnasium-compliant Train Game Environment for Reinforcement Learning
import random
import numpy as np
from gymnasium import spaces

class TrainGameEnv:
    """
    A Gymnasium-compatible environment for train capacity management.
    
    The agent must manage train capacity while maximizing passenger boarding
    and minimizing waste (unused capacity) and infrastructure stress.
    
    Action Space:
        - 0: Add carriage (+100 capacity, high cost, high weight)
        - 1: Widen carriage (+50 capacity, medium cost, medium weight)
    
    Observation Space:
        - capacity: Current train capacity [0, inf)
        - passengers_onboard: Current passenger count [0, inf)
        - station_idx: Current station index [0, 12]
        - direction: Travel direction {-1, 1}
        - hour: Current hour [0, 23]
        - minute: Current minute [0, 59]
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 2}
    
    def __init__(self, initial_capacity=50, seed=None, verbose=False, render_mode=None):
        super().__init__()
        
        # Seeding
        self._np_random = None
        self.seed(seed)
        
        # Rendering
        self.render_mode = render_mode
        self.verbose = verbose

        # Game parameters
        self.initial_capacity = initial_capacity
        
        # Define Gymnasium spaces (MUST be attributes, not properties)
        self.action_space = spaces.Discrete(3)  # 0: Add, 1: Widen, 2: No Action
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 12, 1, 23, 59], dtype=np.float32),
            dtype=np.float32
        )

        # Station configuration (LRT-2 line)
        self.stations = [
            "Recto", "Legarda", "Pureza", "V. Mapa", "J. Ruiz", "Gilmore",
            "Betty Go", "Cubao", "Anonas", "Katipunan",
            "Santolan", "Marikina", "Antipolo"
        ]
        self.num_stations = len(self.stations)
        
        # Infrastructure parameters
        self.window_size = 10
        self.base_collapse_threshold = 10.0
        self.max_steps = 2000

        # State variables (initialized in reset())
        self.capacity = None
        self.passengers_onboard = None
        self.station_idx = None
        self.direction = None
        self.sim_minutes = None
        self.steps = None
        self.station_visits = None
        self.previous_onboard = None
        self.weight_window = None
        self.raw_score = None
        self.total_boarded = None
        self.total_unused = None
        self.total_config_cost = None
        self.peak_inefficiency = None
        self.done = None
        self.done_reason = None

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

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

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Returns:
            observation (np.ndarray): Initial state
            info (dict): Additional information
        """
        # Handle seeding
        if seed is not None:
            self.seed(seed)
            
        # Reset all state variables
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
        
        observation = self._get_state()
        info = self._get_info()
        
        return observation, info

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
    
    def _get_info(self):
        """Get additional info dictionary"""
        return {
            'total_boarded': self.total_boarded,
            'total_config_cost': self.total_config_cost,
            'station_visits': self.station_visits,
            'peak_inefficiency': self.peak_inefficiency,
            'current_station': self.stations[self.station_idx] if self.station_idx is not None else None,
            'done_reason': self.done_reason
        }

    def step(self, action):
        """
        Execute one game step with given action.
        
        Returns:
            observation (np.ndarray): New state
            reward (float): Reward for this step
            terminated (bool): Whether episode ended naturally
            truncated (bool): Whether episode was truncated (time limit, etc.)
            info (dict): Additional information
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        alighted_passengers = 0
        self.previous_onboard = self.passengers_onboard
        terminated = False
        truncated = False

        # Process capacity actions
        if action == 0:  # Add carriage (+100 capacity)
            self.capacity += 100
            cost, weight = 2.0, 1.0
        elif action == 1:  # Widen carriage (+50 capacity)  
            self.capacity += 50
            cost, weight = 1.0, 0.5
        elif action == 2:  # No action (do nothing)
            cost, weight = 0.0, 0.0
        else:
            raise ValueError("Invalid action. Valid actions are 0 (Add), 1 (Winden), 2 (No Action).")

        # Track configuration cost (don't subtract from raw_score yet)
        self.total_config_cost += cost

        # Check for infrastructure collapse
        collapse_threshold = max(3.0, self.base_collapse_threshold - (self.steps / 200))
        self.weight_window.append(weight)
        if len(self.weight_window) > self.window_size:
            self.weight_window.pop(0)
        if sum(self.weight_window) >= collapse_threshold:
            self.done = True
            terminated = True
            self.done_reason = f"Collapse at station {self.stations[self.station_idx]}"
            self.raw_score -= 500.0
            info = self._get_info()
            info.update({'alighted': 0, 'collapse': True})
            return self._get_state(), -1000.0, terminated, truncated, info

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

        # Capacity decay: lose 2 capacity every 100 steps (gradual wear and tear)
        if self.steps > 0 and self.steps % 100 == 0:
            self.capacity = max(0, self.capacity - 2)

        unused = max(0, self.capacity - self.passengers_onboard)
        self.peak_inefficiency = max(self.peak_inefficiency, unused)

        # Calculate rewards and penalties
        current_hour = self.sim_minutes // 60
        
        # Reward for boarding passengers
        reward_board = 1.5 * boarded
        
        # Moderate penalty for missed passengers (couldn't board due to capacity)
        missed_passengers = arrivals - boarded
        penalty_missed = 1.0 * missed_passengers  # Balanced penalty for missed opportunities
        
        # Penalty for unused capacity
        penalty_unused = self._calculate_efficiency_penalty(
            unused, alighted_passengers, self.previous_onboard, current_hour
        )
        
        # Penalty for configuration changes (no multiplier - cost is already balanced)
        config_penalty = cost
        
        # BONUS: Perfect capacity matching (95-100% utilization)
        utilization = self.passengers_onboard / max(1, self.capacity)
        capacity_match_bonus = 0.0
        
        # Rush hour bonus (extra reward for good management during peak times)
        rush_hour_multiplier = 1.0
        if 6 <= current_hour <= 8 or 17 <= current_hour <= 19:
            rush_hour_multiplier = 2.0  # Double bonus during rush hour
        elif 11 <= current_hour <= 13:
            rush_hour_multiplier = 1.5  # 1.5x bonus during lunch rush
        
        if 0.95 <= utilization <= 1.0:
            # Perfect match: near-full capacity
            capacity_match_bonus = 10.0 * rush_hour_multiplier
        elif 0.85 <= utilization < 0.95:
            # Good match: high utilization
            capacity_match_bonus = 5.0 * rush_hour_multiplier
        elif 0.70 <= utilization < 0.85:
            # Decent match: moderate utilization
            capacity_match_bonus = 2.0 * rush_hour_multiplier
        
        # Calculate step reward
        step_reward = reward_board - penalty_unused - config_penalty - penalty_missed + capacity_match_bonus

        # Update game state
        self.raw_score += step_reward
        self.total_boarded += boarded
        self.total_unused += unused
        self.station_visits += 1
        self.steps += 1

        # Advance simulation time
        self.sim_minutes += 5
        if self.sim_minutes >= 22 * 60:
            self.done = True
            terminated = True
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
            truncated = True
            self.done_reason = "Max steps reached."

        # Prepare info dictionary (consistent structure)
        info = self._get_info()
        info.update({
            'alighted': alighted_passengers,
            'boarded': boarded,
            'arrivals': arrivals,
            'penalty_unused': penalty_unused,
            'config_penalty': config_penalty,
            'efficiency_ratio': alighted_passengers / max(1, self.previous_onboard),
            'step_reward': step_reward,
            'utilization': utilization,
            'capacity_match_bonus': capacity_match_bonus,
            'rush_hour_multiplier': rush_hour_multiplier
        })

        # Return standard Gymnasium 5-tuple (FIX: removed double cost subtraction)
        return self._get_state(), step_reward, terminated, truncated, info

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

    def render(self):
        """Render the environment for visualization"""
        if self.render_mode == 'human':
            draw_train(self)

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