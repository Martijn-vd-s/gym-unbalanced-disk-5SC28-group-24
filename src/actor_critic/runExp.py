from ppo import PPOTrainer
import time
import matplotlib.pyplot as plt
import os
# Force PyTorch to use 1 thread to prevent fighting with Python multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

from UnbalancedDiskExp import UnbalancedDisk_exp_sincos

ENV_CLS = UnbalancedDisk_exp_sincos

ENV_KWARGS = dict(umax=3.0, dt=0.025)

trainer = PPOTrainer(
    env_cls=ENV_CLS,
    env_kwargs=ENV_KWARGS,
    n_envs=4,        
    n_steps=256,      
    ppo_epochs=10,    # Takes 10 training steps per data batch!
    lr_actor=1e-4,
    ent_coef=0.01,    
    total_steps=500_000, 
)

trainer.load("ppo_best_tracking.pth")

n_steps = 500
  # Number of steps to run the demo episode for

if __name__ == "__main__":
    env = ENV_CLS(**ENV_KWARGS)
    obs, _ = env.reset()
    trainer.actor.eval()

    thetas, refs, rewards, omegas, voltages = [], [], [], [], []

    print("\n=== DEMO — close the pygame window or wait for it to finish ===")
    print(
        f"{'Step':>6}  {'θ (deg)':>9}  {'θ_ref (deg)':>11}  {'err (deg)':>9}  {'ω (rad/s)':>10}  {'V':>6}  {'reward':>8}"
    )
    print("-" * 75)

    with torch.no_grad():
        for step in range(n_steps):
            sin_th = obs[0]
            cos_th = obs[1]
            omega = obs[2]
            theta = np.arctan2(sin_th, cos_th)
            # Default to pi if env doesn't have th_ref explicitly set, needed for tracking later on
            theta_ref = getattr(env, "th_ref", np.pi) 

            err = ((theta - theta_ref + np.pi) % (2 * np.pi)) - np.pi

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mu = trainer.actor(obs_t)
            action = mu.item() 

            obs, reward, term, trunc, _ = env.step(action)

            # Track data for plots
            thetas.append(theta)
            refs.append(theta_ref)
            rewards.append(reward)
            omegas.append(omega)
            voltages.append(action)

            if step % 10 == 0:
                print(
                    f"{step:>6}  {np.rad2deg(theta):>9.2f}  "
                    f"{np.rad2deg(theta_ref):>11.2f}  "
                    f"{np.rad2deg(err):>9.2f}  "
                    f"{omega:>10.3f}  {action:>+6.3f}  {reward:>8.4f}"
                )

            env.render()
            # time.sleep(0.022)

            if term or trunc:
                obs, _ = env.reset()

    env.close()
    trainer.actor.train()

    # time series plots
    t = np.arange(len(thetas)) * ENV_KWARGS.get("dt", 0.025)

    fig, ax = plt.subplots(figsize=(12, 4))

    thetas_plot = np.unwrap(thetas)
    refs_plot = np.unwrap(refs)
    ax.plot(t, np.rad2deg(refs_plot), "--", color="#888", lw=1.5, label="θ_ref")
    ax.plot(t, np.rad2deg(thetas_plot), color="#4c8cbf", lw=1.5, label="θ actual")
    ax.set_ylabel("Angle [deg]")
    ax.set_xlabel("Time [s]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Demo Trajectory")

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
