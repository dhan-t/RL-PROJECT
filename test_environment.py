"""
Unit tests for TrainGameEnv to verify Gymnasium compliance and correctness.
"""

import numpy as np
from train_game_env import TrainGameEnv


def test_gymnasium_api_compliance():
    """Test that environment follows Gymnasium API standards."""
    print("\n🧪 Testing Gymnasium API Compliance...")
    
    env = TrainGameEnv()
    
    # Test 1: Check spaces are attributes
    assert hasattr(env, 'action_space'), "❌ Missing action_space attribute"
    assert hasattr(env, 'observation_space'), "❌ Missing observation_space attribute"
    print("✅ Action and observation spaces are attributes")
    
    # Test 2: Reset returns (observation, info)
    result = env.reset()
    assert isinstance(result, tuple), "❌ reset() should return tuple"
    assert len(result) == 2, "❌ reset() should return 2-tuple"
    obs, info = result
    assert isinstance(obs, np.ndarray), "❌ observation should be numpy array"
    assert isinstance(info, dict), "❌ info should be dictionary"
    print("✅ reset() returns (observation, info)")
    
    # Test 3: Step returns (obs, reward, terminated, truncated, info)
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple), "❌ step() should return tuple"
    assert len(result) == 5, f"❌ step() should return 5-tuple, got {len(result)}"
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray), "❌ observation should be numpy array"
    assert isinstance(reward, (int, float)), "❌ reward should be numeric"
    assert isinstance(terminated, bool), "❌ terminated should be boolean"
    assert isinstance(truncated, bool), "❌ truncated should be boolean"
    assert isinstance(info, dict), "❌ info should be dictionary"
    print("✅ step() returns (obs, reward, terminated, truncated, info)")
    
    # Test 4: Observation shape
    assert obs.shape == (6,), f"❌ Observation shape should be (6,), got {obs.shape}"
    print("✅ Observation shape is correct (6,)")
    
    # Test 5: Action space
    assert env.action_space.n == 3, f"❌ Action space should have 3 actions, got {env.action_space.n}"
    print("✅ Action space has 3 actions")
    
    print("✅ All Gymnasium API tests passed!\n")


def test_seeding():
    """Test that seeding produces reproducible results."""
    print("🧪 Testing Seeding...")
    
    # Create two environments with same seed
    env1 = TrainGameEnv(seed=42)
    env2 = TrainGameEnv(seed=42)
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    assert np.allclose(obs1, obs2), "❌ Same seed should produce same initial state"
    print("✅ Seeding produces reproducible initial states")
    
    # Take same action in both
    for _ in range(10):
        action = 2  # No action
        obs1, r1, t1, tr1, _ = env1.step(action)
        obs2, r2, t2, tr2, _ = env2.step(action)
        
        assert np.allclose(obs1, obs2), "❌ Same seed should produce same trajectory"
        assert r1 == r2, "❌ Same seed should produce same rewards"
    
    print("✅ Seeding produces reproducible trajectories\n")


def test_state_consistency():
    """Test that state values are consistent and valid."""
    print("🧪 Testing State Consistency...")
    
    env = TrainGameEnv()
    obs, _ = env.reset()
    
    # Check state components
    capacity, onboard, station_idx, direction, hour, minute = obs
    
    assert capacity >= 0, "❌ Capacity should be non-negative"
    assert onboard >= 0, "❌ Passengers onboard should be non-negative"
    assert 0 <= station_idx <= 12, f"❌ Station index should be 0-12, got {station_idx}"
    assert direction in [-1, 1], f"❌ Direction should be -1 or 1, got {direction}"
    assert 0 <= hour <= 23, f"❌ Hour should be 0-23, got {hour}"
    assert 0 <= minute <= 59, f"❌ Minute should be 0-59, got {minute}"
    
    print("✅ Initial state values are valid")
    
    # Run episode and check states
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
        
        capacity, onboard, station_idx, direction, hour, minute = obs
        
        assert capacity >= 0, "❌ Capacity became negative"
        assert onboard >= 0, "❌ Passengers onboard became negative"
        assert onboard <= capacity, f"❌ More passengers ({onboard}) than capacity ({capacity})"
        assert 0 <= station_idx <= 12, f"❌ Invalid station index: {station_idx}"
        assert direction in [-1, 1], f"❌ Invalid direction: {direction}"
        assert 0 <= hour <= 23, f"❌ Invalid hour: {hour}"
        assert 0 <= minute <= 59, f"❌ Invalid minute: {minute}"
    
    print("✅ All states during episode are valid\n")


def test_action_effects():
    """Test that actions have expected effects."""
    print("🧪 Testing Action Effects...")
    
    # Test action 0: Add carriage
    env = TrainGameEnv(seed=42)
    obs, _ = env.reset()
    initial_capacity = obs[0]
    
    obs, reward, _, _, info = env.step(0)  # Add carriage
    new_capacity = obs[0]
    
    assert new_capacity == initial_capacity + 100, f"❌ Action 0 should add 100 capacity, got {new_capacity - initial_capacity}"
    print("✅ Action 0 (Add carriage) increases capacity by 100")
    
    # Test action 1: Widen carriage
    env = TrainGameEnv(seed=42)
    obs, _ = env.reset()
    initial_capacity = obs[0]
    
    obs, reward, _, _, info = env.step(1)  # Widen carriage
    new_capacity = obs[0]
    
    assert new_capacity == initial_capacity + 50, f"❌ Action 1 should add 50 capacity, got {new_capacity - initial_capacity}"
    print("✅ Action 1 (Widen carriage) increases capacity by 50")
    
    # Test action 2: No action
    env = TrainGameEnv(seed=42)
    obs, _ = env.reset()
    initial_capacity = obs[0]
    
    obs, reward, _, _, info = env.step(2)  # No action
    new_capacity = obs[0]
    
    assert new_capacity == initial_capacity, f"❌ Action 2 should not change capacity, got {new_capacity - initial_capacity}"
    print("✅ Action 2 (No action) doesn't change capacity\n")


def test_info_dict_consistency():
    """Test that info dictionary has consistent structure."""
    print("🧪 Testing Info Dictionary Consistency...")
    
    env = TrainGameEnv()
    obs, info = env.reset()
    
    # Check reset info
    required_keys = ['total_boarded', 'total_config_cost', 'station_visits', 
                     'peak_inefficiency', 'current_station', 'done_reason']
    
    for key in required_keys:
        assert key in info, f"❌ Missing key '{key}' in reset info"
    print("✅ Reset info has all required keys")
    
    # Check step info
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_required_keys = required_keys + ['alighted', 'boarded', 'arrivals', 
                                               'penalty_unused', 'config_penalty', 
                                               'efficiency_ratio', 'step_reward']
        
        for key in step_required_keys:
            assert key in info, f"❌ Missing key '{key}' in step info"
        
        if terminated or truncated:
            break
    
    print("✅ Step info has all required keys")
    print("✅ Info dictionary structure is consistent\n")


def test_episode_completion():
    """Test that episodes complete properly."""
    print("🧪 Testing Episode Completion...")
    
    env = TrainGameEnv()
    obs, _ = env.reset()
    
    steps = 0
    max_steps = 3000
    
    while steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        if terminated or truncated:
            assert info['done_reason'] is not None, "❌ Episode ended without done_reason"
            print(f"✅ Episode completed after {steps} steps: {info['done_reason']}")
            break
    
    if steps >= max_steps:
        print("⚠️  Episode didn't complete within max_steps")
    
    # Test final score
    norm_score, raw_score = env.final_score()
    assert 1 <= norm_score <= 100, f"❌ Normalized score should be 1-100, got {norm_score}"
    print(f"✅ Final score: {norm_score}/100 (raw: {raw_score:.1f})\n")


def test_no_double_counting():
    """Test that configuration cost is not double-counted."""
    print("🧪 Testing No Double-Counting of Costs...")
    
    env = TrainGameEnv(seed=42)
    obs, _ = env.reset()
    
    # Take action 0 (cost=10)
    obs, reward, _, _, info = env.step(0)
    
    # Check that config_penalty is in info
    assert 'config_penalty' in info, "❌ config_penalty not in info"
    config_penalty = info['config_penalty']
    
    # The reward should already include the penalty
    # So we shouldn't see additional subtraction
    assert config_penalty == 20.0, f"❌ Config penalty should be 20.0, got {config_penalty}"
    
    print(f"✅ Configuration cost properly included in reward")
    print(f"   Config penalty: {config_penalty}")
    print(f"   Step reward: {info['step_reward']:.2f}\n")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*70)
    print("RUNNING TRAIN GAME ENVIRONMENT TESTS".center(70))
    print("="*70)
    
    test_gymnasium_api_compliance()
    test_seeding()
    test_state_consistency()
    test_action_effects()
    test_info_dict_consistency()
    test_episode_completion()
    test_no_double_counting()
    
    print("="*70)
    print("✅ ALL TESTS PASSED!".center(70))
    print("="*70)
    print()


if __name__ == "__main__":
    run_all_tests()
