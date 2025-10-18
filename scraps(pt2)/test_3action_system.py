#!/usr/bin/env python3
"""
Quick test to verify the 3-action system is working correctly.
Run this before retraining to ensure all components are compatible.
"""

from train_game_env import TrainGameEnv
import sys

def test_environment():
    """Test that environment supports all 3 actions"""
    print("="*70)
    print("TESTING ENVIRONMENT - 3-ACTION SYSTEM")
    print("="*70)
    
    env = TrainGameEnv()
    
    # Check action space
    print(f"\n✓ Action space: {env.action_space}")
    assert env.action_space.n == 3, f"❌ Expected 3 actions, got {env.action_space.n}"
    print("  ✅ Correct: 3 actions (Add, Widen, No Action)")
    
    # Test all actions
    obs, info = env.reset()
    print(f"\n✓ Initial state: {obs}")
    
    actions = {
        0: "Add Carriage",
        1: "Widen Carriage", 
        2: "No Action"
    }
    
    for action, name in actions.items():
        env_test = TrainGameEnv()
        env_test.reset()
        
        try:
            next_obs, reward, terminated, truncated, info = env_test.step(action)
            print(f"  ✅ Action {action} ({name}): reward={reward:.2f}")
        except Exception as e:
            print(f"  ❌ Action {action} ({name}): FAILED - {e}")
            return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Environment is ready!")
    print("="*70)
    print("\nNext steps:")
    print("1. Delete old saved agents: rm saved_agents/*.pt saved_agents/*.pkl")
    print("2. Open rl_training.ipynb")
    print("3. Run cells 3-6 to retrain agents with 3-action system")
    print("4. Expected Actor-Critic score: 85-95/100")
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
