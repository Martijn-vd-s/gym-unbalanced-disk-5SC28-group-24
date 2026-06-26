"""
Run a trained DQN policy on the REAL unbalanced-disk hardware.
DQN analogue of runExp.py: state -> QNet -> argmax over discrete voltages.
"""
import time, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gymnasium import spaces
from UnbalancedDiskExp import UnbalancedDisk_exp

# ===================== MUST MATCH TRAINING =====================
# >>> Paste the 6 voltages from UnbalancedDiskDiscrete.py, same order. <
TRAIN_ACTION_MAP = [-3.0, -1.0, -0.2, 0.2, 1.0, 3.0]   # <-- EDIT THIS

# Encoding flags. These MUST equal what encode_obs() used during training.
# Your current checkpoint trained with OFFSET = 0.0. Only change if you retrain.
OMEGA_OFFSET = 0.0     # set to 1.874 ONLY if you retrain with it on
OMEGA_SCALE  = 8.0

CHECKPOINT = "DQN_swingup_v1.pt"
ENV_KWARGS = dict(umax=3.0, dt=0.025)
N_STEPS    = 1500
TOP_TOL    = 0.30
# ==============================================================


class QNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        layers, last = [], state_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]; last = h
        layers += [nn.Linear(last, n_actions)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def encode_obs(obs):
    """Raw [theta, omega] -> [sin th, cos th, (omega+OFFSET)/SCALE].
    MUST be identical to training."""
    th = float(obs[0]); om = float(obs[1])
    return np.array([np.sin(th), np.cos(th), (om + OMEGA_OFFSET) / OMEGA_SCALE],
                    dtype=np.float32)


def load_qnet(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    net = QNet(int(cfg["state_dim"]), int(cfg["n_actions"]), tuple(cfg["hidden"])).to(device)
    net.load_state_dict(ckpt["model"]); net.eval()
    return net, cfg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f"Error: checkpoint '{CHECKPOINT}' not found.")
    net, cfg = load_qnet(CHECKPOINT, device)
    n_actions = int(cfg["n_actions"])
    print(f"Loaded '{CHECKPOINT}': state_dim={cfg['state_dim']}, n_actions={n_actions}, "
          f"hidden={tuple(cfg['hidden'])}")

    # Safety checks
    assert len(TRAIN_ACTION_MAP) == n_actions, (
        f"TRAIN_ACTION_MAP has {len(TRAIN_ACTION_MAP)} entries, network expects "
        f"{n_actions}. Paste the correct map from UnbalancedDiskDiscrete.py.")
    ckpt_scale = cfg.get("omega_scale", OMEGA_SCALE)
    if abs(ckpt_scale - OMEGA_SCALE) > 1e-9:
        print(f"WARNING: OMEGA_SCALE {OMEGA_SCALE} != checkpoint {ckpt_scale}.")

    # Build hardware env, override action map to the trained one
    env = UnbalancedDisk_exp(**ENV_KWARGS)
    env.num_actions = n_actions
    env.discrete_action_map = list(TRAIN_ACTION_MAP)
    env.action_space = spaces.Discrete(n_actions)
    print("Action map ->", env.discrete_action_map)
    print(f"Encoding -> [sin, cos, (omega + {OMEGA_OFFSET}) / {OMEGA_SCALE}]")

    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs

    thetas, refs, rewards, omegas, voltages = [], [], [], [], []
    top_steps = 0

    print("\n=== DEMO - close the pygame window or wait until it finishes ===")
    print(f"{'Step':>6}  {'theta(deg)':>10}  {'ref(deg)':>9}  {'err(deg)':>9}  "
          f"{'omega':>9}  {'V':>6}  {'reward':>8}")
    print("-" * 72)

    for step in range(N_STEPS):
        theta = env.unwrapped.th
        omega = env.unwrapped.omega
        theta_ref = getattr(env.unwrapped, "th_ref", np.pi)
        err = ((theta - theta_ref + np.pi) % (2 * np.pi)) - np.pi

        with torch.no_grad():
            s = torch.as_tensor(encode_obs([theta, omega]), device=device).unsqueeze(0)
            action = int(torch.argmax(net(s), dim=1).item())

        voltage = env.unwrapped.discrete_action_map[action]

        out = env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, _ = out
        else:
            obs, reward, term, _ = out; trunc = False

        thetas.append(theta); refs.append(theta_ref); rewards.append(reward)
        omegas.append(omega); voltages.append(voltage)
        if abs(((theta - np.pi + np.pi) % (2 * np.pi)) - np.pi) < TOP_TOL:
            top_steps += 1

        if step % 10 == 0:
            print(f"{step:>6}  {np.rad2deg(theta):>10.2f}  {np.rad2deg(theta_ref):>9.2f}  "
                  f"{np.rad2deg(err):>9.2f}  {omega:>9.3f}  {voltage:>+6.3f}  {reward:>8.4f}")

        env.render()
        if term or trunc:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs

    env.close()

    # ----- Plots (same layout as runExp.py) -----
    t = np.arange(len(thetas)) * ENV_KWARGS["dt"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, np.rad2deg(refs), "--", color="#888", lw=1.5, label="theta_ref")
    axes[0].plot(t, np.rad2deg(thetas), color="#4c8cbf", lw=1.5, label="theta actual")
    axes[0].set_ylabel("Angle [deg]"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Demo Trajectory (DQN)")
    axes[1].plot(t, omegas, color="#e07b39", lw=1.2)
    axes[1].axhline(0, color="#888", lw=0.8, ls="--")
    axes[1].set_ylabel("omega [rad/s]"); axes[1].grid(alpha=0.3)
    axes[2].plot(t, voltages, color="#8172b2", lw=1.2)
    for y in (3.0, -3.0): axes[2].axhline(y, color="#888", lw=0.8, ls="--")
    axes[2].axhline(0.0, color="#888", lw=0.8, ls=":")
    axes[2].set_ylabel("Voltage [V]"); axes[2].set_ylim(-3.5, 3.5); axes[2].grid(alpha=0.3)
    axes[3].plot(t, rewards, color="#5ba85e", lw=1.2)
    axes[3].set_ylabel("Reward"); axes[3].set_xlabel("Time [s]"); axes[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("demo_trajectory_DQN.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thetas, omegas, color="#7B85FF", lw=1.5, alpha=0.8, label="Trajectory")
    ax.plot(thetas[0], omegas[0], 'go', markersize=8, label="Start")
    ax.plot(thetas[-1], omegas[-1], 'r*', markersize=12, label="End")
    ax.plot(refs[0] if refs else np.pi, 0, 'g^', markersize=10, label="Target (pi, 0)")
    ax.set_xlabel("Theta (rad)"); ax.set_ylabel("Omega (rad/s)")
    ax.set_title("Demo Episode: Phase Diagram (DQN)")
    ax.grid(alpha=0.4, ls='--'); ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout(); plt.savefig("demo_phase_diagram_DQN.png", dpi=150, bbox_inches="tight"); plt.close()

    errs = np.array([((th - r + np.pi) % (2*np.pi) - np.pi) for th, r in zip(thetas, refs)])
    print("\nPlots saved -> demo_trajectory_DQN.png AND demo_phase_diagram_DQN.png")
    print(f"Mean reward : {np.mean(rewards):.4f}")
    print(f"RMSE        : {np.rad2deg(np.sqrt(np.mean(errs**2))):.2f} deg")
    print(f"Time at top : {100*top_steps/max(len(thetas),1):.1f}%")
    print(f"Max |V|     : {np.max(np.abs(voltages)):.3f} V")
    print(f"Mean |omega|: {np.mean(np.abs(omegas)):.3f} rad/s")