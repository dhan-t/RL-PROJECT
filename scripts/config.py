"""
Configuration settings for RL training pipeline.
"""

# Training parameters
DEFAULT_EPISODES = 1000
DEFAULT_SEED = 42

# Agent hyperparameters
MONTE_CARLO_CONFIG = {
    'eps': 0.1,
    'gamma': 0.99,
    'eps_decay': 0.995,
    'eps_min': 0.01
}

Q_LEARNING_CONFIG = {
    'alpha': 0.1,
    'gamma': 0.99,
    'eps': 0.1,
    'eps_decay': 0.995,
    'eps_min': 0.01,
    'alpha_decay': 0.9999,
    'alpha_min': 0.01
}

# --- CHANGES ARE HERE ---
ACTOR_CRITIC_CONFIG = {
    'state_dim': 6,
    'action_dim': 3,
    'lr': 1e-4,          # <-- Lowered learning rate for more stable updates
    'gamma': 0.99,
    'entropy_coef': 0.005 # <-- Reduced to encourage less random action (exploitation)
}

# Environment parameters
N_ACTIONS = 3
ACTION_NAMES = ['Add Carriage', 'Widen Carriage', 'No Action']

# File paths
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent

# File paths
SAVED_AGENTS_DIR = BASE_DIR / 'saved_agents'
RESULTS_DIR = BASE_DIR / 'results'