# gui_train_game.py
from train_game_env import TrainGameEnv  # keep original import (your env file)
import os
import random
import threading
import time
import pickle
import tkinter as tk
from tkinter import messagebox, ttk


import numpy as np
import torch
import torch.nn as nn

# New: reproducibility constants
DEFAULT_SEED = 42
# seconds between agent steps (slower UI but deterministic)
AGENT_STEP_DELAY = 0.3

# -------------------------
# Agent loading system (auto-loads on import)
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_agents")
# name -> agent-like object exposing .policy(state, greedy=False)
LOADED_AGENTS = {}

# Normalization used at training time (keep consistent with rl_training.ipynb)


def _normalize_state_for_agent(state):
    cap, onboard, station_idx, direction, hour, minute = state
    return np.array([
        cap / 1000.0,
        onboard / 500.0,
        station_idx / 12.0,
        (direction + 1) / 2.0,
        hour / 23.0,
        minute / 59.0
    ], dtype=np.float32)

# Recreate the network architecture used in training so we can load state_dict


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=3, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        policy = torch.softmax(self.actor(x), dim=-1)  # probabilities
        value = self.critic(x)
        return policy, value

# ActorCritic wrapper that exposes policy(state, greedy=False) -> int action

class ActorCriticAgentWrapper:
    def __init__(self, model_path, state_dim=6, action_dim=3, device=None):
        self.device = device or (torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(
            ckpt, dict) else ckpt
        # try to infer dims
        sdim = ckpt.get("state_dim", state_dim) if isinstance(
            ckpt, dict) else state_dim
        adim = action_dim
        try:
            if isinstance(state_dict, dict):
                # some checkpoints might store 'actor.weight' or 'actor.weight'
                # infer action dim from actor weight row
                if "actor.weight" in state_dict:
                    adim = state_dict["actor.weight"].shape[0]
        except Exception:
            pass
        self.net = ActorCriticNet(
            state_dim=sdim, action_dim=adim).to(self.device)
        # try to load directly, else try stripping common prefixes
        try:
            self.net.load_state_dict(state_dict)
        except Exception:
            # try to strip "net." or "module." prefixes
            new_sd = {}
            try:
                for k, v in state_dict.items():
                    nk = k
                    if k.startswith("net."):
                        nk = nk[len("net."):]
                    if k.startswith("module."):
                        nk = nk[len("module."):]
                    new_sd[nk] = v
                self.net.load_state_dict(new_sd)
            except Exception as e:
                raise RuntimeError(f"Failed to load AC checkpoint: {e}")
        self.net.eval()

    def policy(self, state, greedy=False):
        st = _normalize_state_for_agent(state)
        t = torch.tensor(st, dtype=torch.float32,
                         device=self.net.fc1.weight.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.net(t)
        probs = probs.cpu().squeeze(0)
        if greedy:
            return int(torch.argmax(probs).item())
        dist = torch.distributions.Categorical(probs)
        return int(dist.sample().item())

# Tabular agent pickle loader


def _load_tabular_agent_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Simple discretizer and stubs used so pickles saved from notebooks can be unpickled


def discretize_state(state):
    cap, onboard, station_idx, direction, hour, minute = state
    cap_bin = min(int(cap // 100), 20)
    on_bin = min(int(onboard // 50), 10)
    dir_bin = 1 if direction >= 0 else 0
    time_minutes = hour * 60 + minute
    operating_start = 4 * 60
    minutes_since_start = max(0, time_minutes - operating_start)
    time_period = min(int(minutes_since_start // 180), 5)
    return (cap_bin, on_bin, int(station_idx), dir_bin, time_period)


class MonteCarloAgent:
    """Stub for unpickling MonteCarloAgent instances."""

    def __init__(self, n_actions=3, eps=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.eps = eps
        self.gamma = gamma
        self.Q = {}
        self.returns = {}

    def policy(self, state, greedy=False):
        ds = discretize_state(state)
        qvals = [self.Q.get((ds, a), 0.0) for a in range(self.n_actions)]
        if (not greedy) and (hasattr(self, "eps") and random.random() < self.eps):
            return random.randint(0, self.n_actions - 1)
        if all(q == 0 for q in qvals):
            return 2  # Prefer no action when no Q-values
        return int(np.argmax(qvals))


class QLearningAgent:
    """Stub for unpickling QLearningAgent instances."""

    def __init__(self, n_actions=3, alpha=0.1, gamma=0.99, eps=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = {}

    def policy(self, state, greedy=False):
        ds = discretize_state(state)
        qvals = [self.Q.get((ds, a), 0.0) for a in range(self.n_actions)]
        if (not greedy) and (hasattr(self, "eps") and random.random() < self.eps):
            return random.randint(0, self.n_actions - 1)
        if all(q == 0 for q in qvals):
            return 2  # Prefer no action when no Q-values
        return int(np.argmax(qvals))

# Simple deterministic fallback agent


class SimpleRuleAgent:
    def __init__(self):
        self.n_actions = 3

    def select_action(self, state, greedy=True):
        cap, onboard, *_ = state
        cap = max(1.0, float(cap))
        load = float(onboard) / cap
        if load > 0.85:
            return 0  # Add carriage
        if load > 0.5:
            return 1  # Widen carriage
        return 2  # No action (default)

    def policy(self, state, greedy=False):
        return self.select_action(state, greedy=greedy)

# Load saved agents into LOADED_AGENTS


def load_saved_agents(save_dir=SAVE_DIR):
    global LOADED_AGENTS
    LOADED_AGENTS = {}

    if not os.path.isdir(save_dir):
        print(f"🔴 Saved agents dir not found: {save_dir}")
        # register fallback
        LOADED_AGENTS["SimpleRule"] = SimpleRuleAgent()
        print("ℹ️ Registered SimpleRule fallback agent.")
        return LOADED_AGENTS

    # Actor-Critic (try common filenames)
    ac_candidates = ["actor_critic_best_model.pt",
        "actor_critic.pt", "actor.pt"]
    for ac_name in ac_candidates:
        ac_path = os.path.join(save_dir, ac_name)
        if os.path.isfile(ac_path):
            try:
                LOADED_AGENTS["Actor-Critic"] = ActorCriticAgentWrapper(
                    ac_path)
                print(f"✅ Loaded Actor-Critic from {ac_path}")
                break
            except Exception as e:
                print(f"⚠️ Failed to load Actor-Critic from {ac_path}: {e}")

    # Load tabular pickles (.pkl / .pickle)
    for fname in os.listdir(save_dir):
        lower = fname.lower()
        p = os.path.join(save_dir, fname)
        if not os.path.isfile(p):
            continue
        try:
            if lower.endswith(".pkl") or lower.endswith(".pickle"):
                agent = _load_tabular_agent_pickle(p)
                if agent is not None:
                    key = os.path.splitext(fname)[0]
                    LOADED_AGENTS[key] = agent
                    print(f"✅ Loaded tabular agent '{key}' from {p}")
        except Exception as e:
            print(f"⚠️ Skipping {p}: {e}")

    # If nothing loaded, register fallback
    if not LOADED_AGENTS:
        LOADED_AGENTS["SimpleRule"] = SimpleRuleAgent()
        print("ℹ️ No trained agents found — registered SimpleRule fallback.")

    return LOADED_AGENTS


def get_loaded_agent(name):
    return LOADED_AGENTS.get(name)


# Auto-load at import time
try:
    load_saved_agents()
except Exception as e:
    print(f"⚠️ Error during auto-loading agents: {e}")
    LOADED_AGENTS["SimpleRule"] = SimpleRuleAgent()

# -------------------------
# GUI + Game Logic (unchanged, except agent_play uses get_loaded_agent first)
# -------------------------


class TrainGameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚆 Train Game")
        self.root.geometry("900x600")
        self.root.config(bg="#202020")

        # Initialize env with fixed seed for reproducibility (matches CLI runs)
        try:
            self.env = TrainGameEnv(seed=DEFAULT_SEED)
        except TypeError:
            # If env doesn't accept `seed` kwarg, create normally and seed RNGs globally
            self.env = TrainGameEnv()
            random.seed(DEFAULT_SEED)
            np.random.seed(DEFAULT_SEED)
            torch.manual_seed(DEFAULT_SEED)

        self.state, _ = self.env.reset()
        self.running = False
        self.training_thread = None
        self.stop_training_flag = threading.Event()

        self.show_home_screen()

    # ========================
    # SCREEN HELPERS
    # ========================
    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
            
    def show_leaderboards(self):
        # You can add leaderboard logic here later
        self.clear_screen()
        tk.Label(self.root, text="Leaderboards (Coming Soon)", font=("Arial Black", 24), bg="#202020", fg="white").pack(pady=60)
        tk.Button(self.root, text="Back", font=("Arial", 16), command=self.show_home_screen).pack(pady=30)

    def show_home_screen(self):
        self.clear_screen()
        title = tk.Label(self.root, text="🚆 TRAIN GAME", font=(
            "Arial Black", 30), bg="#202020", fg="white")
        title.pack(pady=60)

        btn_play = tk.Button(self.root, text="Play", font=(
            "Arial", 20), width=15, command=self.show_instructions)
        btn_play.pack(pady=15)

        btn_agent = tk.Button(self.root, text="Agent Play", font=(
            "Arial", 20), width=15, command=self.choose_agent_screen)
        btn_agent.pack(pady=15)

        btn_lab = tk.Button(self.root, text="Live Training Lab", font=(
            "Arial", 20), width=15, command=self.show_live_training_lab)
        btn_lab.pack(pady=15)

        btn_leaderboard = tk.Button(self.root, text="Leaderboards", font=(
            "Arial", 20), width=15, command=self.show_leaderboards)
        btn_leaderboard.pack(pady=15)

        lbl = tk.Label(
            self.root,
            text=(
                "📜 Instructions:\n\n"
                "Manage the train’s capacity to maximize passengers without collapsing!\n"
                "Each station move = 1 step.\n\n"
                "Actions:\n"
                "  [Add Carriage] → +100 capacity\n"
                "  [Widen Carriage] → +50 capacity\n\n"
                "Click anywhere to start."
            ),
            font=("Arial", 14),
            justify="left",
            bg="#202020",
            fg="white",
        )
        lbl.pack(padx=30, pady=40)
        self.root.bind("<Button-1>", lambda e: self.start_game())

    def show_live_training_lab(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.clear_screen()
        self.stop_training_flag.clear()

        # Main frame
        main_frame = tk.Frame(self.root, bg="#202020")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls
        control_panel = tk.Frame(main_frame, bg="#2c2c2c", width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right panel for visualization
        vis_panel = tk.Frame(main_frame, bg="#202020")
        vis_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Control Panel Widgets ---
        tk.Label(control_panel, text="🚀 Live Training Lab", font=("Arial Black", 16), bg="#2c2c2c", fg="white").pack(pady=10)

        # Agent selection
        tk.Label(control_panel, text="Agent:", font=("Arial", 12), bg="#2c2c2c", fg="white").pack(pady=(10,0))
        self.agent_var = tk.StringVar(value='Q-Learning')
        agent_menu = ttk.Combobox(control_panel, textvariable=self.agent_var, values=['Q-Learning', 'Actor-Critic'])
        agent_menu.pack(pady=5)

        # Hyperparameter sliders
        self.hyperparams = {}
        self.slider_frame = tk.Frame(control_panel, bg="#2c2c2c")
        self.slider_frame.pack(pady=20, fill=tk.X, padx=5)

        def create_slider(parent, name, from_, to, resolution, default):
            frame = tk.Frame(parent, bg="#2c2c2c")
            label = tk.Label(frame, text=name, font=("Arial", 10), bg="#2c2c2c", fg="white")
            label.pack()
            var = tk.DoubleVar(value=default)
            slider = tk.Scale(frame, variable=var, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL, bg="#2c2c2c", fg="white", troughcolor="#555")
            slider.pack(fill=tk.X)
            self.hyperparams[name] = var
            return frame

        # Sliders for Q-Learning
        self.q_sliders = [
            create_slider(self.slider_frame, "Epsilon", 0.0, 1.0, 0.01, 0.1),
            create_slider(self.slider_frame, "Alpha", 0.01, 1.0, 0.01, 0.1),
            create_slider(self.slider_frame, "Gamma", 0.8, 1.0, 0.01, 0.99)
        ]

        # Sliders for Actor-Critic
        self.ac_sliders = [
            create_slider(self.slider_frame, "Learning Rate", 1e-5, 1e-2, 1e-5, 3e-4),
            create_slider(self.slider_frame, "Gamma", 0.8, 1.0, 0.01, 0.99),
            create_slider(self.slider_frame, "Entropy Coef", 0.0, 0.1, 0.001, 0.01)
        ]

        def on_agent_change(*args):
            agent = self.agent_var.get()
            for slider in self.q_sliders + self.ac_sliders:
                slider.pack_forget()
            if agent == 'Q-Learning':
                for slider in self.q_sliders:
                    slider.pack(pady=5)
            else:
                for slider in self.ac_sliders:
                    slider.pack(pady=5)
        
        self.agent_var.trace("w", on_agent_change)
        on_agent_change() # Initial setup

        # Start/Stop buttons
        self.start_button = tk.Button(control_panel, text="Start Training", font=("Arial", 14), command=self.start_live_training)
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(control_panel, text="Stop Training", font=("Arial", 14), command=self.stop_live_training, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        tk.Button(control_panel, text="Back to Menu", font=("Arial", 12), command=self.return_home).pack(side=tk.BOTTOM, pady=20)

        # --- Visualization Panel Widgets ---
        # Matplotlib chart
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("Live Score Over Episodes")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Score")
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], 'r-')

        self.canvas_plt = FigureCanvasTkAgg(self.fig, master=vis_panel)
        self.canvas_plt.get_tk_widget().pack(pady=10)

        # Game canvas
        self.canvas = tk.Canvas(vis_panel, width=700, height=120, bg="#303030", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.station_coords = []
        for i, st in enumerate(self.env.stations):
            x = 80 + i * 80
            self.canvas.create_oval(x - 5, 80 - 5, x + 5, 80 + 5, fill="white")
            self.canvas.create_text(x, 100, text=st, fill="white", font=("Arial", 10))
            self.station_coords.append(x)
        self.train = self.canvas.create_rectangle(70, 60, 90, 75, fill="cyan")

    def start_live_training(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_training_flag.clear()

        self.training_thread = threading.Thread(target=self.live_training_loop, daemon=True)
        self.training_thread.start()

    def stop_live_training(self):
        self.stop_training_flag.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def live_training_loop(self):
        agent_name = self.agent_var.get()
        
        # Instantiate agent
        if agent_name == 'Q-Learning':
            from scripts.agents import QLearningAgent
            agent = QLearningAgent(n_actions=self.env.action_space.n)
        elif agent_name == 'Actor-Critic':
            from scripts.agents import ActorCriticAgent
            agent = ActorCriticAgent()
        else:
            return

        scores = []
        episodes = []

        for ep in range(10000): # Effectively infinite loop
            if self.stop_training_flag.is_set():
                break

            # Update hyperparameters from GUI at the start of each episode
            if agent_name == 'Q-Learning':
                agent.eps = self.hyperparams["Epsilon"].get()
                agent.alpha = self.hyperparams["Alpha"].get()
                agent.gamma = self.hyperparams["Gamma"].get()
            elif agent_name == 'Actor-Critic':
                # Note: Changing LR mid-training requires re-creating the optimizer
                for g in agent.optimizer.param_groups:
                    g['lr'] = self.hyperparams["Learning Rate"].get()
                agent.gamma = self.hyperparams["Gamma"].get()
                agent.entropy_coef = self.hyperparams["Entropy Coef"].get()

            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            
            # Simplified training loop for live visualization
            trajectory = []
            while not done:
                if isinstance(agent, ActorCriticAgent):
                    action, log_prob, value = agent.policy(state)
                    next_state, reward, term, trunc, info = self.env.step(action)
                    trajectory.append((state, (action, log_prob, value), reward, next_state, term, trunc))
                else: # Q-Learning
                    action = agent.policy(state)
                    next_state, reward, term, trunc, info = self.env.step(action)
                    agent.update(state, action, reward, next_state, term, trunc)

                state = next_state
                ep_reward += reward
                done = term or trunc

                # Update game canvas
                self.root.after(0, self.move_train)
                time.sleep(0.01) # Small delay to allow UI to update

            if isinstance(agent, ActorCriticAgent):
                agent.learn(trajectory)

            final_score, _ = self.env.final_score()
            scores.append(final_score)
            episodes.append(ep)

            # Update plot
            self.line.set_data(episodes, scores)
            self.ax.relim()
            self.ax.autoscale_view()
            self.root.after(0, self.canvas_plt.draw)

        print("Live training stopped.")

    def show_instructions(self):
        pass

    def start_game(self):
        self.root.unbind("<Button-1>")
        self.clear_screen()
        self.running = True
        self.env.reset()
        self.build_game_ui()
        self.update_ui()

    def build_game_ui(self):
        self.title = tk.Label(self.root, text="🚆 Train Capacity Manager", font=(
            "Arial Black", 22), bg="#202020", fg="white")
        self.title.pack(pady=20)

        self.info_label = tk.Label(self.root, text="", font=(
            "Consolas", 14), bg="#202020", fg="white", justify="left")
        self.info_label.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=700,
                                height=120, bg="#303030", highlightthickness=0)
        self.canvas.pack(pady=20)

        # Station markers
        self.station_coords = []
        for i, st in enumerate(self.env.stations):
            x = 80 + i * 80
            self.canvas.create_oval(x - 5, 80 - 5, x + 5, 80 + 5, fill="white")
            self.canvas.create_text(
                x, 100, text=st, fill="white", font=("Arial", 10))
            self.station_coords.append(x)

        # Train rectangle
        self.train = self.canvas.create_rectangle(
            70, 60, 90, 75, fill="lightblue")

        btn_frame = tk.Frame(self.root, bg="#202020")
        btn_frame.pack(pady=20)

        btn_add = tk.Button(btn_frame, text="Add Carriage (+100)",
                            font=("Arial", 16), width=20, command=lambda: self.take_action(0))
        btn_add.grid(row=0, column=0, padx=10)

        btn_widen = tk.Button(btn_frame, text="Widen Carriage (+50)",
                              font=("Arial", 16), width=20, command=lambda: self.take_action(1))
        btn_widen.grid(row=0, column=1, padx=10)

        btn_quit = tk.Button(self.root, text="Quit to Menu", font=(
            "Arial", 14), command=self.return_home)
        btn_quit.pack(pady=30)

    def move_train(self):
        """Animate train to next station position."""
        idx = self.env.station_idx
        if 0 <= idx < len(self.station_coords):
            target_x = self.station_coords[idx]
            current_coords = self.canvas.coords(self.train)
            current_x = (current_coords[0] + current_coords[2]) / 2
            step = 4 if target_x > current_x else -4
            while abs(target_x - current_x) > 5:
                self.canvas.move(self.train, step, 0)
                self.root.update()
                time.sleep(0.02)
                current_coords = self.canvas.coords(self.train)
                current_x = (current_coords[0] + current_coords[2]) / 2
            # Snap to exact
            dx = target_x - current_x
            self.canvas.move(self.train, dx, 0)
            self.root.update()

    def take_action(self, action):
        if not self.running:
            return
        state, reward, term, trunc, info = self.env.step(action)
        self.move_train()
        self.state = state
        self.update_ui()
        if term or trunc:
            self.running = False
            score, raw = self.env.final_score()
            messagebox.showinfo(
                "Game Over", f"{info['done_reason']}\n\nFinal Score: {score}")
            self.return_home()

    def update_ui(self):
        s = self.env
        info_text = (
            f"📍 Station: {s.stations[s.station_idx]}\n"
            f"🚋 Capacity: {s.capacity}\n"
            f"👥 Onboard: {s.passengers_onboard}\n"
            f"🕒 Time: {s.sim_minutes//60:02d}:{s.sim_minutes % 60:02d}\n"
            f"➡ Direction: {'Eastbound' if s.direction == 1 else 'Westbound'}\n"
            f"🏗 Config Cost: {s.total_config_cost:.1f}\n"
            f"🎯 Total Boarded: {s.total_boarded}\n"
            f"⚠️ Peak Inefficiency: {s.peak_inefficiency}\n"
            f"💯 Score (raw): {s.raw_score:.1f}"
        )
        self.info_label.config(text=info_text)

    def return_home(self):
        self.stop_live_training()
        try:
            self.env.close()
        except Exception:
            pass
        self.show_home_screen()

    # ========================
    # AGENT PLAY
    # ========================
    def choose_agent_screen(self):
        self.clear_screen()
        tk.Label(self.root, text="🤖 Choose an Agent", font=(
            "Arial Black", 24), bg="#202020", fg="white").pack(pady=40)
        btns = [
            ("Monte Carlo", "monte_carlo_agent.pkl"),
            ("Q-Learning", "q_learning_agent.pkl"),
            ("Actor-Critic", "actor_critic_best_model.pt"),
        ]
        for name, file in btns:
            tk.Button(
                self.root,
                text=name,
                font=("Arial", 18),
                width=18,
                command=lambda f=file, n=name: self.agent_play(f, n),
            ).pack(pady=10)
        tk.Button(self.root, text="Back", font=("Arial", 16),
                  command=self.show_home_screen).pack(pady=30)

    def agent_play(self, filename, agent_name):
        self.clear_screen()
        tk.Label(self.root, text=f"🤖 {agent_name} Playing...", font=(
            "Arial", 20), bg="#202020", fg="white").pack(pady=20)

        # Main info area (same stats as player screen)
        info_frame = tk.Frame(self.root, bg="#202020")
        info_frame.pack(pady=6)
        agent_info_label = tk.Label(info_frame, text="", font=(
            "Consolas", 12), bg="#202020", fg="white", justify="left")
        agent_info_label.pack()

        # Canvas for train / stations (same as player)
        self.canvas = tk.Canvas(self.root, width=700,
                                height=120, bg="#303030", highlightthickness=0)
        self.canvas.pack(pady=10)

        # Station markers & coords
        self.station_coords = []
        for i, st in enumerate(self.env.stations):
            x = 80 + i * 80
            self.canvas.create_oval(x - 5, 80 - 5, x + 5, 80 + 5, fill="white")
            self.canvas.create_text(
                x, 100, text=st, fill="white", font=("Arial", 10))
            self.station_coords.append(x)
        self.train = self.canvas.create_rectangle(
            70, 60, 90, 75, fill="lightgreen")

        # Right-side panel: action history list and controls
        right_frame = tk.Frame(self.root, bg="#202020")
        right_frame.pack(pady=8)

        tk.Label(right_frame, text="Agent Action History", font=(
            "Arial", 14), bg="#202020", fg="white").pack(pady=(0, 6))
        actions_frame = tk.Frame(right_frame, bg="#202020")
        actions_frame.pack()

        actions_scroll = tk.Scrollbar(actions_frame, orient=tk.VERTICAL)
        actions_listbox = tk.Listbox(
            actions_frame, width=40, height=12, yscrollcommand=actions_scroll.set, font=("Consolas", 10))
        actions_scroll.config(command=actions_listbox.yview)
        actions_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        actions_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Return button
        tk.Button(self.root, text="Back to Menu", font=("Arial", 14),
                  command=self.show_home_screen).pack(pady=12)

        action_names = {0: "Add Carriage (+100)", 1: "Widen Carriage (+50)"}

        def update_agent_ui(env_obj, info=None, action_taken=None):
            # Build same stats text as update_ui uses
            s = env_obj
            info_text = (
                f"📍 Station: {s.stations[s.station_idx]}\n"
                f"🚋 Capacity: {s.capacity}\n"
                f"👥 Onboard: {s.passengers_onboard}\n"
                f"🕒 Time: {s.sim_minutes//60:02d}:{s.sim_minutes % 60:02d}\n"
                f"➡ Direction: {'Eastbound' if s.direction == 1 else 'Westbound'}\n"
                f"🏗 Config Cost: {s.total_config_cost:.1f}\n"
                f"🎯 Total Boarded: {s.total_boarded}\n"
                f"⚠️ Peak Inefficiency: {s.peak_inefficiency}\n"
                f"💯 Score (raw): {s.raw_score:.1f}"
            )
            agent_info_label.config(text=info_text)
            if action_taken is not None:
                ts = time.strftime("%H:%M:%S")
                actions_listbox.insert(
                    tk.END, f"[{ts}] {action_names.get(action_taken, str(action_taken))}")
                actions_listbox.yview_moveto(1.0)

            # sync main app train position so move_train animates correctly
            try:
                self.env.station_idx = env_obj.station_idx
            except Exception:
                pass

        def run_agent():
            # Ensure deterministic RNGs for this run (match CLI notebook)
            random.seed(DEFAULT_SEED)
            np.random.seed(DEFAULT_SEED)
            torch.manual_seed(DEFAULT_SEED)

            # Create a fresh, seeded environment for agent evaluation
            try:
                env = TrainGameEnv(seed=DEFAULT_SEED)
            except TypeError:
                env = TrainGameEnv()
            state, _ = env.reset()

               # initial UI update
            self.root.after(0, lambda: update_agent_ui(
                   env, action_taken=None))

                # get / build policy (same heuristics as before)
            agent_obj = None
            agent_obj = get_loaded_agent(agent_name) or get_loaded_agent(
                    agent_name.replace(" ", "-"))
            if agent_obj is None:
                    lf = filename.lower()
                    if "actor" in lf:
                        agent_obj = get_loaded_agent("Actor-Critic")
                    elif "q" in lf:
                        agent_obj = get_loaded_agent(
                            "Q-Learning") or get_loaded_agent("q_learning_agent") or get_loaded_agent("q_learning")
                    elif "monte" in lf or "mc" in lf:
                        agent_obj = get_loaded_agent("Monte Carlo") or get_loaded_agent(
                            "monte_carlo_agent") or get_loaded_agent("monte_carlo")

            # Force neural agents into eval mode (disable dropout/noise)
        
            try:
                if hasattr(agent_obj, "net"):
                    agent_obj.net.eval()
            except Exception:
                pass

            # Always use greedy policy for evaluation to be deterministic
            policy = None
            if agent_obj is not None:
                    def policy(s):
                        # Normalize/greedy handled by wrappers/policies; enforce greedy everywhere
                        try:
                            return int(agent_obj.policy(s, greedy=True))
                        except Exception:
                            if hasattr(agent_obj, "select_action"):
                                return int(agent_obj.select_action(s, greedy=True))
                            raise

            else:
                    # fallback disk loading (unchanged)
                    try:
                        saved_path = os.path.join(SAVE_DIR, filename)
                        if filename.endswith(".pt") and os.path.isfile(saved_path):
                            wc = ActorCriticAgentWrapper(saved_path)
                            agent_obj_local = wc
                            # enforce greedy eval
                            agent_obj_local.net.eval()
                            def policy(s): return int(
                                agent_obj_local.policy(s, greedy=True))
                        else:
                            p = saved_path
                            if os.path.isfile(p):
                                with open(p, "rb") as f:
                                    pick = pickle.load(f)
                                if hasattr(pick, "policy"):
                                    def policy(s): return int(
                                        pick.policy(s, greedy=True))
                                elif hasattr(pick, "select_action"):
                                    def policy(s): return int(
                                        pick.select_action(s, greedy=True))
                                else:
                                    raise RuntimeError(
                                        "Pickle loaded but no policy/select_action found.")
                            else:
                                raise FileNotFoundError(
                                    f"Agent file not found: {saved_path}")
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror(
                            "Error", f"Failed to load agent: {e}"))
                        self.root.after(0, self.show_home_screen)
                        return

            # Run deterministic evaluation loop mirroring CLI timing/logic
            done = False
            step_idx = 0
            total_reward_acc = 0.0
            print(
                f"\n=== AGENT RUN START: {agent_name} (seed={DEFAULT_SEED}) ===")
            while not done:
                try:
                    action = policy(state)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Agent policy error: {e}"))
                    break

                # perform step
                state, reward, term, trunc, info = env.step(action)
                step_idx += 1
                total_reward_acc += float(reward)

                # schedule UI update + animation on main thread
                self.root.after(0, lambda env_obj=env, act=action: update_agent_ui(
                    env_obj, action_taken=act))
                self.root.after(0, self.move_train)

                done = term or trunc

                # CLI-style log per step for reproducibility comparison
                try:
                    boarded = info.get("boarded", 0)
                    alighted = info.get("alighted", 0)
                    eff = info.get("efficiency_ratio", 0)
                except Exception:
                    boarded = alighted = eff = 0
                print(f"Step {step_idx:03d} | Action: {action_names.get(action, action)} | Reward: {reward:.2f} | Boarded: {boarded} | Alighted: {alighted} | Capacity: {env.capacity} | Eff: {eff}")

                time.sleep(AGENT_STEP_DELAY)

            score, raw = env.final_score()
            # CLI-style summary (same as notebook)
            print("\n" + "="*60)
            print(f"AGENT FINISHED: {agent_name}")
            print(f"Final Score: {score} / 100")
            print(f"Raw Score: {raw:.2f}")
            print(f"Total Reward (accumulated): {total_reward_acc:.2f}")
            print("="*60 + "\n")

            self.root.after(0, lambda: messagebox.showinfo(
                "Agent Finished", f"{agent_name} Score: {score}"))
            self.root.after(0, self.show_home_screen)

        threading.Thread(target=run_agent, daemon=True).start()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    app = TrainGameApp(root)
    root.mainloop()
