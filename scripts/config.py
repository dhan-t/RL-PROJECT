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

ACTOR_CRITIC_CONFIG = {
    'state_dim': 6,
    'action_dim': 3,
    'lr': 3e-4,
    'gamma': 0.99,
    'entropy_coef': 0.01
}

# Environment parameters
N_ACTIONS = 3
ACTION_NAMES = ['Add Carriage', 'Widen Doors', 'No Action']

# File paths
SAVED_AGENTS_DIR = '../saved_agents'
RESULTS_DIR = '../results'
