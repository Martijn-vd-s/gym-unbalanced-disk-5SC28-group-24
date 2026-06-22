"""
Run a trained Q-learning policy on the REAL unbalanced-disk hardware.

Loads a Q-table (``.pkl``) trained in simulation and runs it greedily on the
physical setup (``UnbalancedDisk_exp``), logging the trajectory and saving
trajectory + phase-diagram plots plus summary statistics.

Consistency requirement: the discretisation (``Discretize_obs``) here MUST be
identical to the one in the training notebook, and the env's action map /
observation range MUST match the training env. Otherwise the loaded Q-table
indexes the wrong states / voltages and the policy behaves randomly.
"""

import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# Q-learning trained on the base 'UnbalancedDisk_exp', NOT the '_sincos' variant.
from UnbalancedDiskExp import UnbalancedDisk_exp


class Discretize_obs(gym.Wrapper):
    """Discretise ``[theta, omega]`` to a state tuple. MUST match the training notebook.

    Angle: NON-uniform grid, fine near the top (phi=0), coarse near the bottom.
    Velocity: uniform over the env's omega range. ``nvec=[<ignored for angle>, n_omega]``;
    fineness is set by ``fine_w`` / ``fine_res`` / ``coarse_res``.
    """

    def __init__(self, env, nvec, fine_w=0.40, fine_res=0.025, coarse_res=0.10):
        super(Discretize_obs, self).__init__(env)
        o = env.observation_space
        self.omega_low = float(o.low[1]); self.omega_high = float(o.high[1])
        self.n_omega = int(np.array(nvec).flatten()[1])
        # top-centred, non-uniform angle bin edges (fine near 0, coarse toward +-pi)
        pos = list(np.arange(0.0, fine_w, fine_res)) + list(np.arange(fine_w, np.pi, coarse_res)) + [np.pi]
        self.angle_edges = np.array(sorted(set([-e for e in pos] + pos)))
        self.n_angle = len(self.angle_edges) - 1
        self.observation_space = gym.spaces.MultiDiscrete([self.n_angle, self.n_omega])

    def discretize(self, observation):
        """Map continuous ``[theta, omega]`` to integer ``(angle_bin, omega_bin)``."""
        th = float(observation[0]); om = float(observation[1])
        phi = (th % (2 * np.pi)) - np.pi                                  # 0 = top
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
    """Index of the maximum, breaking ties at random."""
    a = np.array(a)
    return np.random.choice(np.arange(len(a), dtype=int)[a == np.max(a)])


# ==========================================
# CONFIGURATION
# ==========================================
ENV_CLS = UnbalancedDisk_exp
ENV_KWARGS = dict(umax=3.0, dt=0.025)

# Trained model to deploy (must be trained with the CURRENT action map / discretisation).
checkpoint_filename = "Q_learning_DATA_v65.pkl"
n_steps = 1500  # number of steps for the demo/evaluation

if __name__ == "__main__":
    # 1. Build the hardware environment
    env_base = ENV_CLS(**ENV_KWARGS)

    # 2. Wrap with the discretiser (identical settings to training)
    nvec_angle = 120
    nvec_rps = 50
    env = Discretize_obs(env_base, nvec=[nvec_angle, nvec_rps])

    # 3. Load the trained Q-table
    print(f"Loading Q-matrix from {checkpoint_filename}...")
    try:
        with open(checkpoint_filename, "rb") as f:
            Qmat = pickle.load(f)
        print(f"Success! Q-matrix loaded with {len(Qmat)} visited states.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    obs, _ = env.reset()

    thetas, refs, rewards, omegas, voltages = [], [], [], [], []

    print("\n=== DEMO - close the pygame window or wait until it finishes ===")
    print(
        f"{'Step':>6}  {'θ (deg)':>9}  {'θ_ref (deg)':>11}  {'err (deg)':>9}  {'ω (rad/s)':>10}  {'V':>6}  {'reward':>8}"
    )
    print("-" * 75)

    for step in range(n_steps):
        # Read the true continuous values straight from the base env for plotting
        theta = env.unwrapped.th
        omega = env.unwrapped.omega
        theta_ref = getattr(env.unwrapped, "th_ref", np.pi)

        err = ((theta - theta_ref + np.pi) % (2 * np.pi)) - np.pi

        # Greedy action: highest Q-value for this discrete state (no exploration)
        q_values = [Qmat[(obs, i)] for i in range(env.action_space.n)]
        action = argmax(q_values)

        # Map the discrete action index back to a voltage (for the plots)
        voltage = env.unwrapped.discrete_action_map[action]

        obs, reward, term, trunc, _ = env.step(action)

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

    # Summary statistics
    errs = np.array([((th - r + np.pi) % (2 * np.pi) - np.pi) for th, r in zip(thetas, refs)])
    print("\nPlots saved -> demo_trajectory.png AND demo_phase_diagram.png")
    print(f"Mean reward : {np.mean(rewards):.4f}")
    print(f"RMSE        : {np.rad2deg(np.sqrt(np.mean(errs**2))):.2f}°")
    print(f"Max |V|     : {np.max(np.abs(voltages)):.3f} V")
    print(f"Mean |ω|    : {np.mean(np.abs(omegas)):.3f} rad/s")
