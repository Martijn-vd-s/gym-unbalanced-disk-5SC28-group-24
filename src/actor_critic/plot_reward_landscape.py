"""
3D reward landscape for the UnbalancedDisk reward function.
Axes: angle error vs angular velocity (u fixed at 0, so no control penalties).
The crimson ridge shows the target-omega curve that guides swing-up.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Parameters (must match UnbalancedDisk._reward) ──────────────────────────
sigma_err   = np.pi / 4.0
sigma_track = np.deg2rad(7.0)
A           = 12.0
sigma_swing = 2.0

# ── Grid ─────────────────────────────────────────────────────────────────────
N = 250
err   = np.linspace(-np.pi, np.pi, N)
omega = np.linspace(-20, 20, N)
ERR, OMEGA = np.meshgrid(err, omega)

# ── Reward components (u = 0, prev_u = 0 → no penalties) ────────────────────
r_balance    = np.exp(-(ERR**2) / (2 * sigma_err**2))
target_omega = A * np.sin(ERR / 2.0)
r_swing      = 0.5 * np.exp(-((OMEGA - target_omega)**2) / (2 * sigma_swing**2))
r_track      = 1.5 * np.exp(-(ERR**2) / (2 * sigma_track**2))
R_total      = r_balance + r_swing + r_track

# ── Optimal-velocity ridge (where r_swing is maximised) ──────────────────────
err_ridge   = np.linspace(-np.pi, np.pi, 600)
omega_ridge = A * np.sin(err_ridge / 2.0)
r_ridge = (
    np.exp(-(err_ridge**2) / (2 * sigma_err**2))
    + 0.5                                                     # r_swing peak
    + 1.5 * np.exp(-(err_ridge**2) / (2 * sigma_track**2))
)

# ── Figure: two panels ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))

# ── Left: top-down contour view ───────────────────────────────────────────────
ax_top = fig.add_subplot(121)
cf = ax_top.contourf(np.rad2deg(ERR), OMEGA, R_total, levels=60, cmap="viridis")
ax_top.contour(np.rad2deg(ERR), OMEGA, R_total, levels=12,
               colors="white", linewidths=0.4, alpha=0.4)
ax_top.plot(np.rad2deg(err_ridge), omega_ridge,
            color="crimson", lw=2.2, label=r"target $\omega = A\,\sin(\epsilon/2)$")
ax_top.scatter([0], [0], color="white", edgecolors="black", s=80, zorder=5)
fig.colorbar(cf, ax=ax_top, label="Reward")
ax_top.set_xlabel("Error  [deg]")
ax_top.set_ylabel(r"$\omega$  [rad/s]")
ax_top.set_title("Top-down view", fontsize=12)
ax_top.legend(fontsize=9)

# ── Right: angled 3D surface ──────────────────────────────────────────────────
ax_3d = fig.add_subplot(122, projection="3d")
surf = ax_3d.plot_surface(
    np.rad2deg(ERR), OMEGA, R_total,
    cmap="viridis", alpha=0.85,
    linewidth=0, antialiased=True,
    rstride=2, cstride=2,
)
ax_3d.plot(np.rad2deg(err_ridge), omega_ridge, r_ridge,
           color="crimson", lw=2.5, zorder=10,
           label=r"target $\omega = A\,\sin(\epsilon/2)$")
ax_3d.scatter([0], [0], [float(R_total[N // 2, N // 2])],
              color="white", edgecolors="black", s=80, zorder=11)
fig.colorbar(surf, ax=ax_3d, shrink=0.45, aspect=12, pad=0.08, label="Reward")
ax_3d.set_xlabel("Error  [deg]", labelpad=10)
ax_3d.set_ylabel(r"$\omega$  [rad/s]", labelpad=10)
ax_3d.set_zlabel("Reward", labelpad=8)
ax_3d.set_title("Angled view", fontsize=12)
ax_3d.view_init(elev=28, azim=-55)
ax_3d.set_box_aspect([1.6, 1.2, 1.0])
ax_3d.legend(loc="upper right", fontsize=9)

fig.suptitle("Reward landscape  (u = 0)", fontsize=14, y=1.01)
plt.tight_layout()
out = "reward_landscape_3d.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
