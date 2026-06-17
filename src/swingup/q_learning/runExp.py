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
    MOET exact gelijk zijn aan Discretize_obs in de training-notebook!
    Hoek: NIET-uniform -> fijn vlak bij de top (phi=0), grof bij de bodem. Snelheid: uniform.
    nvec=[<genegeerd voor hoek>, n_omega].  fine_w / fine_res / coarse_res bepalen de fijnheid.
    """
    def __init__(self, env, nvec, fine_w=0.40, fine_res=0.025, coarse_res=0.10):
        super(Discretize_obs, self).__init__(env)
        o = env.observation_space
        self.omega_low  = float(o.low[1]);  self.omega_high = float(o.high[1])
        self.n_omega = int(np.array(nvec).flatten()[1])
        pos = list(np.arange(0.0, fine_w, fine_res)) + list(np.arange(fine_w, np.pi, coarse_res)) + [np.pi]
        self.angle_edges = np.array(sorted(set([-e for e in pos] + pos)))
        self.n_angle = len(self.angle_edges) - 1
        self.observation_space = gym.spaces.MultiDiscrete([self.n_angle, self.n_omega])

    def discretize(self, observation):
        th = float(observation[0]); om = float(observation[1])
        phi = (th % (2 * np.pi)) - np.pi
        a = int(np.clip(np.searchsorted(self.angle_edges, phi, side='right') - 1, 0, self.n_angle - 1))
        om = np.clip(om, self.omega_low, self.omega_high)
        b = int(np.clip((om - self.omega_low) / (self.omega_high - self.omega_low) * self.n_omega, 0, self.n_omega - 1))
        return (a, b)

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
checkpoint_filename = "Q_learning_DATA_v46.pkl"
n_steps = 1500  # Aantal stappen voor de evaluatie/demo

if __name__ == "__main__":
    # 1. Maak de base environment
    env_base = ENV_CLS(**ENV_KWARGS)
    
    # 2. Wikkel de environment om de observaties te discretiseren (precies zoals in je Jupyter Notebook)
    nvec_angle = 120
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