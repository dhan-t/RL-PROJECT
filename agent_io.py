import os
import pickle
import torch
from tkinter import messagebox

# ==========================================
# LOAD FUNCTIONS (mirror training behavior)
# ==========================================

def load_actor_critic_agent(filepath, state_dim=6, action_dim=3):
    """Load a saved Actor-Critic agent"""
    from rl_training import ActorCriticAgent  # ensure same class definition
    checkpoint = torch.load(filepath, map_location="cpu")
    
    agent = ActorCriticAgent(
        state_dim=checkpoint['state_dim'],
        action_dim=action_dim,
        lr=3e-4,
        gamma=checkpoint['gamma'],
        entropy_coef=checkpoint['entropy_coef']
    )
    
    agent.net.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.net.eval()
    print(f"✅ Loaded Actor-Critic agent from: {filepath}")
    return agent

def load_tabular_agent(filepath):
    """Load a saved tabular agent (Monte Carlo or Q-Learning)"""
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    print(f"✅ Loaded tabular agent from: {filepath}")
    return agent

def load_selected_agent(filename):
    """Unified loader for both .pt and .pkl files"""
    try:
        base_dir = os.path.dirname(__file__)
        filepath = os.path.join(base_dir, "saved_agents", filename)

        if filename.endswith(".pt"):
            agent = load_actor_critic_agent(filepath, state_dim=6, action_dim=3)
            policy = lambda s: int(agent.select_action(torch.tensor(s, dtype=torch.float32))[0])

        elif filename.endswith(".pkl"):
            agent = load_tabular_agent(filepath)
            policy = lambda s: agent.policy(s, greedy=True)

        else:
            raise ValueError("Unsupported agent file type.")
        
        print(f"✅ Loaded agent successfully: {filename}")
        return agent, policy

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load agent: {e}")
        return None, None
