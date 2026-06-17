"""
Standalone tracking evaluation.

Runs the CURRENT trained model (ppo_best.pth) against a *scripted* reference that
actually moves, so you can see whether the policy tracks. No retraining needed.

Run:  python eval_tracking.py
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from UnbalancedDisk import UnbalancedDisk_sincos
from ppo import Actor

DT       = 0.025
N        = 600        # 15 s
SETTLE_S = 3.0        # swing up and settle at the top before we start moving the ref
AMP_DEG  = 15.0
PERIOD_S = 4.0        # square-wave full period


def ref_schedule(t):
    # hold at the top until settled, then step between +-15 deg every PERIOD_S/2
    if t < SETTLE_S:
        return np.pi
    return np.pi + np.deg2rad(AMP_DEG) * np.sign(np.sin(2 * np.pi * (t - SETTLE_S) / PERIOD_S))


def main():
    env = UnbalancedDisk_sincos(umax=3.0, dt=DT, randomise=False)
    obs, _ = env.reset()
    env._random_ref = False                 # WE drive the reference, not the env

    actor = Actor(obs_dim=5, hidden=256)
    ckpt = torch.load("ppo_best.pth", map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    th, ref, omega, volt, errs, t_axis = [], [], [], [], [], []

    with torch.no_grad():
        for k in range(N):
            t = k * DT
            env.th_ref = ref_schedule(t)    # set BEFORE step so this step's obs reflects it

            a = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
            obs, r, term, trunc, _ = env.step(a)

            e = ((env.th - env.th_ref + np.pi) % (2 * np.pi)) - np.pi
            th.append(np.rad2deg(env.th_ref + e))   # true angle, unwrapped to sit near the ref
            ref.append(np.rad2deg(env.th_ref))
            omega.append(env.omega)
            volt.append(float(np.clip(a, -3, 3)))
            errs.append(np.rad2deg(e))
            t_axis.append(t)

            if term or trunc:
                obs, _ = env.reset()
                env._random_ref = False

    errs = np.array(errs)
    mask = np.array(t_axis) >= SETTLE_S
    print(f"Tracking RMSE after settle : {np.sqrt(np.mean(errs[mask] ** 2)):.2f} deg")
    print(f"Max |tracking error|       : {np.max(np.abs(errs[mask])):.2f} deg")
    print(f"Max |V|                    : {np.max(np.abs(volt)):.2f} V")

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(t_axis, ref, "--", color="#888", lw=1.5, label="theta_ref")
    ax[0].plot(t_axis, th, color="#4c8cbf", lw=1.5, label="theta actual")
    ax[0].set_ylabel("Angle [deg]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Reference tracking — scripted +-15 deg square wave")

    ax[1].plot(t_axis, errs, color="#c0392b", lw=1.2)
    ax[1].axhline(0, color="#888", lw=0.8, ls="--")
    ax[1].set_ylabel("Tracking error [deg]"); ax[1].grid(alpha=0.3)

    ax[2].plot(t_axis, volt, color="#8172b2", lw=1.2)
    ax[2].axhline(0, color="#888", lw=0.8, ls=":")
    ax[2].set_ylim(-3.5, 3.5)
    ax[2].set_ylabel("Voltage [V]"); ax[2].set_xlabel("Time [s]"); ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracking_demo.png", dpi=150, bbox_inches="tight")
    print("saved tracking_demo.png")


if __name__ == "__main__":
    main()
