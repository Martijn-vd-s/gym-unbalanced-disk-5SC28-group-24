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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(1)

#########################################################################
#  NETWORKS 
#########################################################################

class Actor(nn.Module):
    """Stable Gaussian policy for PPO"""
    def __init__(self, obs_dim: int = 4, act_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden, act_dim),
            nn.Tanh() 
        )
        # Standalone standard deviation parameter
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5) 

    def forward(self, x):
        h = self.net(x)
        mu = 3.0 * self.mu_head(h) # Scale to +/- 3.0V
        return mu

    def get_action(self, x):
        mu = self(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        action = dist.sample() 
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

class Critic(nn.Module):
    """State-value baseline V(s)."""
    def __init__(self, obs_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


#########################################################################
#  PARALLEL ENVIRONMENT WORKER 
#########################################################################

def _worker(conn, env_cls, env_kwargs):
    sys.path.insert(0, os.getcwd())
    env = env_cls(**env_kwargs)
    step_ctr = 0
    horizon = 500
    obs_raw, _ = env.reset()

    def augment(obs_raw):
        return obs_raw.astype(np.float32)

    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs_raw, reward, term, trunc, _ = env.step(data)
            done = term or trunc
            step_ctr += 1
            if step_ctr >= horizon:
                done = True
            if done:
                obs_raw, _ = env.reset()
                step_ctr = 0
            conn.send((augment(obs_raw), float(reward), done))
        elif cmd == "reset":
            obs_raw, _ = env.reset()
            step_ctr = 0
            conn.send(augment(obs_raw))
        elif cmd == "close":
            env.close()
            break

class ParallelEnvs:
    def __init__(self, n_envs: int, env_cls, env_kwargs: dict):
        self.n = n_envs
        self.parents, self.children = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.procs = [
            mp.Process(target=_worker, args=(c, env_cls, env_kwargs), daemon=True)
            for c in self.children
        ]
        for p in self.procs:
            p.start()

    def reset(self):
        for p in self.parents:
            p.send(("reset", None))
        return np.stack([p.recv() for p in self.parents])

    def step(self, actions):
        for p, a in zip(self.parents, actions):
            p.send(("step", float(a)))
        results = [p.recv() for p in self.parents]
        obs, rew, done = zip(*results)
        return np.stack(obs), np.array(rew, dtype=np.float32), np.array(done)

    def close(self):
        for p in self.parents:
            p.send(("close", None))
        for p in self.procs:
            p.join()


#########################################################################
#  PPO TRAINER
#########################################################################

class PPOTrainer:
    def __init__(self, env_cls, env_kwargs: dict, n_envs=8, n_steps=256,
                 gamma=0.99, lam=0.95, lr_actor=1e-4, lr_critic=1e-3,
                 clip_ratio=0.2, ppo_epochs=10, ent_coef=0.01,
                 total_steps=1_500_000, hidden=256, device="cpu"):
        
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.ent_coef = ent_coef
        self.total_steps = total_steps
        self.device = torch.device(device)

        self.envs = ParallelEnvs(n_envs, env_cls, env_kwargs)
        self.actor = Actor(5, hidden=hidden).to(self.device)
        self.critic = Critic(5, hidden=hidden).to(self.device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.history = dict(ep_returns=[], ep_lengths=[], actor_loss=[], critic_loss=[])

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def train(self):
        obs = self.envs.reset()
        ep_ret_buf = np.zeros(self.n_envs)
        ep_len_buf = np.zeros(self.n_envs, dtype=int)
        
        total_env_steps = 0
        n_updates = self.total_steps // (self.n_envs * self.n_steps)
        best_ret = -np.inf

        pbar = tqdm(range(n_updates), desc="PPO training", unit="update", dynamic_ncols=True)

        for _ in pbar:
            obs_buf, act_buf, rew_buf = [], [], []
            done_buf, val_buf, logp_buf = [], [], []

            # Collect Data using current policy
            for _ in range(self.n_steps):
                obs_t = self._t(obs)
                with torch.no_grad():
                    action, logp = self.actor.get_action(obs_t)
                    value = self.critic(obs_t).squeeze(-1)

                actions_np = action.cpu().numpy()
                clipped_actions = np.clip(actions_np, -3.0, 3.0) 
                next_obs, rew, done = self.envs.step(clipped_actions)

                obs_buf.append(obs)
                act_buf.append(actions_np)
                rew_buf.append(rew)
                done_buf.append(done.astype(np.float32))
                val_buf.append(value.cpu().numpy())
                logp_buf.append(logp.cpu().numpy())

                ep_ret_buf += rew
                ep_len_buf += 1
                for i, d in enumerate(done):
                    if d:
                        self.history["ep_returns"].append(ep_ret_buf[i])
                        self.history["ep_lengths"].append(ep_len_buf[i])
                        ep_ret_buf[i] = 0
                        ep_len_buf[i] = 0

                obs = next_obs
                total_env_steps += self.n_envs

            # Calculate Advantages
            with torch.no_grad():
                last_val = self.critic(self._t(obs)).squeeze(-1).cpu().numpy()

            T = self.n_steps
            adv = np.zeros((T, self.n_envs), dtype=np.float32)
            last_gae = np.zeros(self.n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + self.gamma * next_val * (1 - done_buf[t]) - val_buf[t]
                last_gae = delta + self.gamma * self.lam * (1 - done_buf[t]) * last_gae
                adv[t] = last_gae
            ret = adv + np.array(val_buf)

            # Flatten
            def flat(x): return np.concatenate(x, axis=0)
            obs_f = self._t(flat(obs_buf))
            act_f = self._t(flat(act_buf))
            logp_old_f = self._t(flat(logp_buf))
            adv_f = self._t(adv.flatten())
            ret_f = self._t(ret.flatten())

            # Normalize advantages
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

            # PPO Update (Multiple Epochs over the same data)
            a_losses, c_losses = [], []
            for _ in range(self.ppo_epochs):
                # Recalculate probabilities with current network
                mu = self.actor(obs_f)
                std = self.actor.log_std.exp().expand_as(mu)
                dist = Normal(mu, std)
                logp = dist.log_prob(act_f).sum(-1)
                entropy = dist.entropy().mean()

                # PPO Clipped Surrogate Objective
                ratio = torch.exp(logp - logp_old_f)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_f
                actor_loss = -(torch.min(ratio * adv_f, clip_adv)).mean() - self.ent_coef * entropy

                # Critic Update
                val_pred = self.critic(obs_f).squeeze(-1)
                critic_loss = nn.functional.mse_loss(val_pred, ret_f)

                # Optimizer Steps
                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_critic.step()

                a_losses.append(actor_loss.item())
                c_losses.append(critic_loss.item())

            self.history["actor_loss"].append(np.mean(a_losses))
            self.history["critic_loss"].append(np.mean(c_losses))

            recent_rets = self.history["ep_returns"][-20:] or [0.0]
            mean_ret = np.mean(recent_rets)

            if mean_ret > best_ret:
                best_ret = mean_ret
                self.save("ppo_best.pth")

            pbar.set_postfix({
                "steps": f"{total_env_steps / 1e3:.1f}k",
                "ret": f"{mean_ret:.1f}",
                "best": f"{best_ret:.1f}",
                "c_loss": f"{np.mean(c_losses):.2f}"
            })

        self.envs.close()
        np.savez("ppo_training_history.npz",
                 **{k: np.array(v) for k, v in self.history.items()})
        return self.history

    def save(self, path="ppo_model.pth"):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path="ppo_model.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


#########################################################################
#  DEMO
#########################################################################

def demo(trainer, env_cls, env_kwargs: dict, n_steps: int = 500):
    """Greedy rollout with live rendering, terminal prints, and full plotting."""
    import time
    import matplotlib.pyplot as plt

    env = env_cls(**env_kwargs)
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

            # Track data for plots — display angle in reference frame to avoid ±180° jumps
            thetas.append(theta_ref + err)
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
            time.sleep(env.dt)

            if term or trunc:
                obs, _ = env.reset()

    env.close()
    trainer.actor.train()

    # time series plots
    t = np.arange(len(thetas)) * env_kwargs.get("dt", 0.025)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, np.rad2deg(refs), "--", color="#888", lw=1.5, label="θ_ref")
    ax.plot(t, np.rad2deg(thetas), color="#4c8cbf", lw=1.5, label="θ actual")
    ax.set_ylabel("Angle [deg]")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Demo Trajectory")

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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.path.insert(0, os.getcwd())
    
    # import enviorment 
    from actor_critic.results.UnbalancedDisk import UnbalancedDisk_sincos
    ENV_CLS = UnbalancedDisk_sincos

    ENV_KWARGS = dict(umax=3.0, dt=0.025, randomise=True)

    trainer = PPOTrainer(
        env_cls=ENV_CLS,
        env_kwargs=ENV_KWARGS,
        n_envs=8,        
        n_steps=256,      
        ppo_epochs=10,    # Takes 10 training steps per data batch!
        lr_actor=1e-4,
        ent_coef=0.01,    
        total_steps=1_500_000, 
    )

    trainer.train()
    trainer.load("ppo_best.pth")
    ENV_KWARGS = dict(umax=3.0, dt=0.025, randomise=True)
    demo(trainer, ENV_CLS, ENV_KWARGS, n_steps=500)