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
        - 0: Add carriage (+100 capacity)
        - 1: Widen carriage (+50 capacity)
        - 2: No Action
    
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
        
        self.seed(seed)
        self.render_mode = render_mode
        self.verbose = verbose
        self.initial_capacity = initial_capacity
        
        # Define Gymnasium spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 12, 1, 23, 59], dtype=np.float32),
            dtype=np.float32
        )

        # Station configuration
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

        # State variables are initialized in reset()
        self.capacity = None
        self.passengers_onboard = None
        self.station_idx = None
        self.direction = None
        self.sim_minutes = None
        self.steps = None
        self.station_visits = None
        self.weight_window = None
        self.raw_score = None
        self.total_boarded = None
        self.total_config_cost = None
        self.peak_inefficiency = None
        self.done = None
        self.done_reason = None

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

    # ----------------- core simulation methods -----------------
    def _time_multiplier(self, hour):
        if 6 <= hour <= 8:   return 1.9
        if 11 <= hour <= 13: return 1.6
        if 17 <= hour <= 19: return 1.9
        return 0.8 if random.random() < 0.45 else 1.0

    def _arrival_bounds(self, idx):
        if idx in (0, self.num_stations-1, 7):
            return (40, 150)
        return (10, 70)

    def _simulate_arrivals(self):
        amin, amax = self._arrival_bounds(self.station_idx)
        base = random.randint(amin, amax)
        current_hour = self.sim_minutes // 60
        mult = self._time_multiplier(current_hour)
        surge_factor = 1.0 + (self.steps / self.max_steps) * 2.0
        return max(0, int(round(base * mult * surge_factor)))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        self.capacity = self.initial_capacity
        self.passengers_onboard = 0
        self.raw_score = 0.0
        self.total_boarded = 0
        self.total_config_cost = 0.0
        self.peak_inefficiency = 0
        self.station_idx = 0
        self.direction = +1
        self.weight_window = []
        self.sim_minutes = random.randint(4 * 60, 22 * 60 - 1)
        self.steps = 0
        self.station_visits = 0
        self.done = False
        self.done_reason = None
        
        return self._get_state(), self._get_info()

    def _get_state(self):
        hour = self.sim_minutes // 60
        minute = self.sim_minutes % 60
        return np.array([
            float(self.capacity), float(self.passengers_onboard), float(self.station_idx),
            float(self.direction), float(hour), float(minute)
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            'total_boarded': self.total_boarded,
            'total_config_cost': self.total_config_cost,
            'station_visits': self.station_visits,
            'peak_inefficiency': self.peak_inefficiency,
            'current_station': self.stations[self.station_idx],
            'done_reason': self.done_reason
        }

    # --- THIS IS THE CORRECTED AND RE-INDENTED STEP FUNCTION ---
    def step(self, action):
        """
        Execute one game step with a simplified, more direct reward signal.
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        terminated = False
        truncated = False

        # 1. PROCESS ACTIONS AND IMMEDIATE COSTS
        if action == 0:  # Add carriage
            self.capacity += 100
            config_cost = 25.0
        elif action == 1:  # Widen carriage
            self.capacity += 50
            config_cost = 12.0
        elif action == 2:  # No action
            config_cost = 0.0
        else:
            raise ValueError("Invalid action.")

        self.total_config_cost += config_cost

        # Infrastructure collapse check
        weight = config_cost / 12.5
        collapse_threshold = max(3.0, self.base_collapse_threshold - (self.steps / 200))
        self.weight_window.append(weight)
        if len(self.weight_window) > self.window_size:
            self.weight_window.pop(0)
        if sum(self.weight_window) >= collapse_threshold:
            self.done = True
            terminated = True
            self.done_reason = f"Collapse at station {self.stations[self.station_idx]}"
            return self._get_state(), -500.0, terminated, truncated, self._get_info()

        # 2. SIMULATE PASSENGERS
        if self.passengers_onboard > 0:
            alighted_passengers = random.randint(0, self.passengers_onboard)
            self.passengers_onboard -= alighted_passengers
        if self.station_idx in (0, self.num_stations - 1):
            self.passengers_onboard = 0
            
        arrivals = self._simulate_arrivals()
        space = max(0, self.capacity - self.passengers_onboard)
        boarded = min(arrivals, space)
        self.passengers_onboard += boarded
        missed_passengers = arrivals - boarded
        unused_space = self.capacity - self.passengers_onboard
        self.peak_inefficiency = max(self.peak_inefficiency, unused_space)
        
        # 3. CALCULATE REWARD
        reward_boarded = boarded * 1.0
        penalty_config = config_cost
        penalty_missed = missed_passengers * 0.5
        penalty_unused = unused_space * 0.1
        penalty_time_step = 0.1

        step_reward = reward_boarded - penalty_config - penalty_missed - penalty_unused - penalty_time_step
        
        # 4. UPDATE STATE
        self.raw_score += step_reward
        self.total_boarded += boarded
        self.steps += 1
        self.station_visits += 1
        
        self.sim_minutes += 5
        if self.sim_minutes >= 22 * 60:
            self.done = True; terminated = True; self.done_reason = "End of operating hours"
        
        next_idx = self.station_idx + self.direction
        if next_idx < 0 or next_idx >= self.num_stations:
            self.direction *= -1
            next_idx = self.station_idx + self.direction
        self.station_idx = next_idx
        
        if self.steps >= self.max_steps:
            self.done = True; truncated = True; self.done_reason = "Max steps reached"

        return self._get_state(), step_reward, terminated, truncated, self._get_info()

    def final_score(self):
        """Calculate final normalized score (1-100)"""
        distance_bonus = self.station_visits * 10
        efficiency_penalty = self.peak_inefficiency * 0.1
        config_efficiency = max(0, 1 - (self.total_config_cost / (self.total_boarded + 1)))
        effective_score = (self.raw_score + distance_bonus - efficiency_penalty) * config_efficiency
        
        S_min = -80 * self.station_visits if self.station_visits > 0 else 0
        S_max = 100 * self.station_visits if self.station_visits > 0 else 1
        normalized = round(1 + ((effective_score - S_min) / (S_max - S_min)) * 99)
        
        return max(1, min(100, normalized)), effective_score

    def render(self):
        if self.render_mode == 'human':
            draw_train(self)

    def close(self):
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