"""
Training curve comparison: PPO vs A2C.
Loads ppo_training_history.npz and a2c_training_history.npz produced at the
end of each training run, then saves a report-ready figure.

Run after training both algorithms:
    python plot_training_comparison.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PPO_COLOR = "#4c8cbf"
A2C_COLOR = "#e07b39"
SMOOTH_W  = 50   # moving-average window for loss curves
REWARD_W  = 30   # window for reward smoothing


def moving_avg(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")



def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ── Load histories ────────────────────────────────────────────────────────────
try:
    ppo = load("ppo_training_history.npz")
except FileNotFoundError:
    raise FileNotFoundError("ppo_training_history.npz not found — run ppo.py first.")
try:
    a2c = load("a2c_training_history.npz")
except FileNotFoundError:
    raise FileNotFoundError("a2c_training_history.npz not found — run actor_critic_a2c.py first.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PPO vs A2C — Training Comparison", fontsize=14, fontweight="bold")

# ── Episode returns ───────────────────────────────────────────────────────────
ax = axes[0]
for hist, label, color in [(ppo, "PPO", PPO_COLOR), (a2c, "A2C", A2C_COLOR)]:
    rets = hist["ep_returns"]
    if len(rets) == 0:
        continue
    smooth = moving_avg(rets, REWARD_W)
    n = len(smooth)
    x = np.linspace(0, len(rets), n)
    ax.plot(rets, alpha=0.12, color=color, lw=0.6)
    ax.plot(x, smooth, color=color, lw=2.0, label=label)

ax.set_title("Episode Return")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.legend()
ax.grid(alpha=0.3)

# ── Actor loss ────────────────────────────────────────────────────────────────
ax = axes[1]
for hist, label, color in [(ppo, "PPO", PPO_COLOR), (a2c, "A2C", A2C_COLOR)]:
    losses = hist["actor_loss"]
    if len(losses) == 0:
        continue
    ax.plot(losses, alpha=0.2, color=color, lw=0.7)
    ax.plot(moving_avg(losses, SMOOTH_W), color=color, lw=2.0, label=label)

ax.set_title("Actor Loss  (per update)")
ax.set_xlabel("Update")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)

# ── Critic loss ───────────────────────────────────────────────────────────────
ax = axes[2]
for hist, label, color in [(ppo, "PPO", PPO_COLOR), (a2c, "A2C", A2C_COLOR)]:
    losses = hist["critic_loss"]
    if len(losses) == 0:
        continue
    ax.plot(losses, alpha=0.2, color=color, lw=0.7)
    ax.plot(moving_avg(losses, SMOOTH_W), color=color, lw=2.0, label=label)

ax.set_title("Critic Loss  (per update)")
ax.set_xlabel("Update")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out = "training_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
