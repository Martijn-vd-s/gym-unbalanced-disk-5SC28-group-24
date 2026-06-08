import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# Importeer de correcte omgeving. 
# Let op: Q-learning trainde op de basis 'UnbalancedDisk_exp', NIET de '_sincos' variant!
from UnbalancedDiskExp import UnbalancedDisk_exp

class Discretize_obs(gym.Wrapper):
    """
    Wrapper om de continue observatieruimte te discretiseren naar MultiDiscrete.
    Dit moet exact overeenkomen met wat er tijdens het trainen is gebruikt!
    """
    def __init__(self, env, nvec):
        super(Discretize_obs, self).__init__(env)
        original_obs_space = env.observation_space
        
        if isinstance(original_obs_space, spaces.Box):
            original_obs_shape = original_obs_space.shape
            self.olow = original_obs_space.low
            self.ohigh = original_obs_space.high
        elif isinstance(original_obs_space, spaces.MultiDiscrete):
            original_obs_shape = original_obs_space.nvec.shape
            self.olow = np.zeros_like(original_obs_space.nvec, dtype=np.float32)
            self.ohigh = (original_obs_space.nvec - 1).astype(np.float32)
        else:
            raise TypeError("Onondersteund observatieruimte type.")

        self.nvec_array = np.array(nvec, dtype=int)
        self.observation_space = gym.spaces.MultiDiscrete(self.nvec_array.flatten())
        self.range_obs = self.ohigh - self.olow
        self.range_obs[self.range_obs == 0] = 1.0

    def discretize(self, observation):
        observation = np.clip(observation, self.olow, self.ohigh)
        discrete_obs_float = ((observation - self.olow) / self.range_obs * self.nvec_array)
        return tuple(np.clip(discrete_obs_float, 0, self.nvec_array - 1).astype(int).flatten())
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.discretize(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.discretize(obs), info

def argmax(a):
    """Breekt eventuele gelijke Q-waardes (ties) willekeurig op."""
    a = np.array(a)
    return np.random.choice(np.arange(len(a), dtype=int)[a==np.max(a)])

# ==========================================
# CONFIGURATIE
# ==========================================
ENV_CLS = UnbalancedDisk_exp
ENV_KWARGS = dict(umax=3.0, dt=0.025)

# Zorg dat dit de naam is van je daadwerkelijk getrainde pickle model
checkpoint_filename = "Q_learning_2_neww_reward_v15.pkl" 
n_steps = 500  # Aantal stappen voor de evaluatie/demo

if __name__ == "__main__":
    # 1. Maak de base environment
    env_base = ENV_CLS(**ENV_KWARGS)
    
    # 2. Wikkel de environment om de observaties te discretiseren (precies zoals in je Jupyter Notebook)
    nvec_angle = 360
    nvec_rps = 50
    env = Discretize_obs(env_base, nvec=[nvec_angle, nvec_rps])
    
    # 3. Laad het getrainde Q-learning model in
    print(f"Q-matrix laden vanuit {checkpoint_filename}...")
    try:
        with open(checkpoint_filename, "rb") as f:
            Qmat = pickle.load(f)
        print(f"Succes! Q-matrix geladen met {len(Qmat)} bezochte states.")
    except Exception as e:
        print(f"Fout bij laden model: {e}")
        exit()

    obs, _ = env.reset()

    thetas, refs, rewards, omegas, voltages = [], [], [], [], []

    print("\n=== DEMO — sluit het pygame scherm of wacht tot dit klaar is ===")
    print(
        f"{'Step':>6}  {'θ (deg)':>9}  {'θ_ref (deg)':>11}  {'err (deg)':>9}  {'ω (rad/s)':>10}  {'V':>6}  {'reward':>8}"
    )
    print("-" * 75)

    for step in range(n_steps):
        # We lezen de werkelijke continue waarden direct uit de base-environment af voor de visualisatie
        theta = env.unwrapped.th
        omega = env.unwrapped.omega
        theta_ref = getattr(env.unwrapped, "th_ref", np.pi) 

        err = ((theta - theta_ref + np.pi) % (2 * np.pi)) - np.pi

        # Vraag alle bekende Q-waardes op voor deze specifieke state-tuple
        q_values = [Qmat[(obs, i)] for i in range(env.action_space.n)]
        
        # Kies de actie met de hoogste Q-waarde (Greedy policy, geen exploration meer)
        action = argmax(q_values)
        # action = 0
        
        # Omdat de actie een discrete index is (bijv. 3), mappen we hem terug naar Voltage voor de plots
        voltage = env.unwrapped.discrete_action_map[action]

        # Voer de stap uit in de omgeving
        obs, reward, term, trunc, _ = env.step(action)

        # Track de data
        thetas.append(theta)
        refs.append(theta_ref)
        rewards.append(reward)
        omegas.append(omega)
        voltages.append(voltage)

        if step % 10 == 0:
            print(
                f"{step:>6}  {np.rad2deg(theta):>9.2f}  "
                f"{np.rad2deg(theta_ref):>11.2f}  "
                f"{np.rad2deg(err):>9.2f}  "
                f"{omega:>10.3f}  {voltage:>+6.3f}  {reward:>8.4f}"
            )

        env.render()
        # time.sleep(env.unwrapped.dt)

        if term or trunc:
            obs, _ = env.reset()

    env.close()

    # ==========================================
    # PLOTS
    # ==========================================
    t = np.arange(len(thetas)) * ENV_KWARGS.get("dt", 0.025)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, np.rad2deg(refs), "--", color="#888", lw=1.5, label="θ_ref")
    ax.plot(t, np.rad2deg(thetas), color="#4c8cbf", lw=1.5, label="θ actual")
    ax.set_ylabel("Angle [deg]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Demo Trajectory (Q-Learning)")

    ax = axes[1]
    ax.plot(t, omegas, color="#e07b39", lw=1.2)
    ax.axhline(0, color="#888", lw=0.8, linestyle="--")
    ax.set_ylabel("ω [rad/s]")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, voltages, color="#8172b2", lw=1.2)
    ax.axhline(3.0, color="#888", lw=0.8, linestyle="--")
    ax.axhline(-3.0, color="#888", lw=0.8, linestyle="--")
    ax.axhline(0.0, color="#888", lw=0.8, linestyle=":")
    ax.set_ylabel("Voltage [V]")
    ax.set_ylim(-3.5, 3.5)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, rewards, color="#5ba85e", lw=1.2)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("demo_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Phase diagram
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(thetas, omegas, color="#7B85FF", lw=1.5, alpha=0.8, label="Trajectory")
    ax.plot(thetas[0], omegas[0], 'go', markersize=8, label="Start")
    ax.plot(thetas[-1], omegas[-1], 'r*', markersize=12, label="End")
    
    target_th = refs[0] if len(refs) > 0 else np.pi
    ax.plot(target_th, 0, 'g^', markersize=10, label="Target (π, 0)")
    
    ax.set_xlabel("Theta (radians)")
    ax.set_ylabel("Omega (rad/s)")
    ax.set_title("Demo Episode: Phase Diagram")
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("demo_phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # summary stats
    errs = np.array([((th - r + np.pi) % (2 * np.pi) - np.pi) for th, r in zip(thetas, refs)])
    print("\nPlots saved -> demo_trajectory.png AND demo_phase_diagram.png")
    print(f"Mean reward : {np.mean(rewards):.4f}")
    print(f"RMSE        : {np.rad2deg(np.sqrt(np.mean(errs**2))):.2f}°")
    print(f"Max |V|     : {np.max(np.abs(voltages)):.3f} V")
    print(f"Mean |ω|    : {np.mean(np.abs(omegas)):.3f} rad/s")