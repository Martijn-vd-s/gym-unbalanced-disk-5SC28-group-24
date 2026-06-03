import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


#########################################################################
#  NETWORKS
#########################################################################


class Actor(nn.Module):
    """Gaussian policy: maps [sin θ, cos θ, ω, err] -> μ, log σ."""

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden: int = 128,
        log_std_min: float = -5,
        log_std_max: float = 1,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def get_action(self, x):
        mu, log_std = self(x)
        dist = Normal(mu, log_std.exp())
        raw = dist.rsample()
        action = 3.0 * torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1 - torch.tanh(raw).pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class Critic(nn.Module):
    """State-value baseline V(s)."""

    def __init__(self, obs_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
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
    """
    Subprocess worker.
    Reward comes directly from the env (set in UnbalancedDisk.py)
    """
    sys.path.insert(0, os.getcwd())
    env = env_cls(**env_kwargs)

    step_ctr = 0
    horizon = 500
    obs_raw, _ = env.reset()

    def augment(obs_raw, env):
        # can be used for testing and debugging the reward function without touching the env code.
        return obs_raw.astype(np.float32)

    # Main loop: receive commands from the main process, step the env, and send back results.
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

            conn.send((augment(obs_raw, env), float(reward), done))

        elif cmd == "reset":
            obs_raw, _ = env.reset()
            step_ctr = 0
            conn.send(augment(obs_raw, env))

        elif cmd == "close":
            env.close()
            break


class ParallelEnvs:
    """Spawn N independent env workers communicating via mp.Pipe.
    Each worker runs an instance of env_cls with env_kwargs.
    Speeds up data collection by parallelizing across CPU cores."""

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
#  A2C TRAINER
#########################################################################


class A2CTrainer:
    """
    Synchronous Advantage Actor-Critic (A2C).
    Reward is taken directly from the environment (UnbalancedDisk.py).
    """

    def __init__(
        self,
        env_cls,
        env_kwargs: dict,
        n_envs: int = 4,
        n_steps: int = 256,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        ent_coef: float = 0.05,
        max_grad_norm: float = 0.5,
        total_steps: int = 2_000_000,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.obs_dim = 3
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.max_grad = max_grad_norm
        self.total_steps = total_steps
        self.device = torch.device(device)

        # Create parallel environments
        self.envs = ParallelEnvs(n_envs, env_cls, env_kwargs)

        # Create actor and critic networks
        self.actor = Actor(self.obs_dim, hidden=hidden).to(self.device)
        self.critic = Critic(self.obs_dim, hidden=hidden).to(self.device)

        # Separate optimizers for actor and critic with different learning rates
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Exponential learning rate decay for stability
        self.sched_actor = optim.lr_scheduler.ExponentialLR(
            self.opt_actor, gamma=0.9999
        )
        self.sched_critic = optim.lr_scheduler.ExponentialLR(
            self.opt_critic, gamma=0.9999
        )

        self.history = dict(
            ep_returns=[],
            ep_lengths=[],
            actor_loss=[],
            critic_loss=[],
            entropy=[],
        )

    def _t(self, x):
        # Convert numpy array to torch tensor on the correct device.
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _gae(self, rewards, values, dones, last_value):
        # Generalized Advantage Estimation (GAE) for computing advantages and returns.
        # GEA is a method to compute advantage estimates that balance bias and variance.
        # It uses a weighted sum of n-step returns, where the weighting is controlled by the λ parameter.
        T = len(rewards)
        advantages = np.zeros((T, self.n_envs), dtype=np.float32)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(values)
        return advantages, returns

    def train(self):
        # Main training loop for A2C.
        # Collects trajectories from parallel environments, computes advantages and returns using GAE, and updates the actor and critic networks.
        obs = self.envs.reset()
        ep_ret_buf = np.zeros(self.n_envs)
        ep_len_buf = np.zeros(self.n_envs, dtype=int)

        total_env_steps = 0
        n_updates = self.total_steps // (self.n_envs * self.n_steps)
        best_ret = -np.inf

        # Progress bar for training updates, showing total environment steps, mean return, best return, episode length, actor loss, critic loss, and entropy.
        pbar = tqdm(
            range(n_updates), desc="A2C training", unit="update", dynamic_ncols=True
        )

        for _ in pbar:
            obs_buf, act_buf, rew_buf = [], [], []
            done_buf, val_buf = [], []

            for _ in range(self.n_steps):
                obs_t = self._t(obs)
                with torch.no_grad():
                    action, _ = self.actor.get_action(obs_t)
                    value = self.critic(obs_t).squeeze(-1)

                actions_np = action.squeeze(-1).cpu().numpy()
                next_obs, rew, done = self.envs.step(actions_np)

                obs_buf.append(obs)
                act_buf.append(action.cpu().numpy())
                rew_buf.append(rew)
                done_buf.append(done.astype(np.float32))
                val_buf.append(value.cpu().numpy())

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

            with torch.no_grad():
                last_val = self.critic(self._t(obs)).squeeze(-1).cpu().numpy()

            advantages, returns = self._gae(rew_buf, val_buf, done_buf, last_val)
            self.sched_actor.step()
            self.sched_critic.step()

            def flat(x):
                # Flatten list of arrays into a single array of shape
                return np.concatenate(x, axis=0)

            # Convert buffers to torch tensors and normalize advantages for stable training.
            obs_f = self._t(flat(obs_buf))
            act_f = self._t(flat(act_buf))
            adv_f = self._t(advantages.flatten())
            ret_f = self._t(returns.flatten())
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

            # Compute actor loss using the log probability of actions under the current policy and the advantages, plus an entropy bonus for exploration.
            # mu is the mean action from the policy, log_std is the log standard deviation, dist is the resulting normal distribution, raw is the pre-tanh action, log_p is the log probability of the action, and entropy is the entropy of the distribution.
            mu, log_std = self.actor(obs_f)
            dist = Normal(mu, log_std.exp())
            raw = torch.atanh((act_f / 3.0).clamp(-0.999, 0.999))
            log_p = (
                dist.log_prob(raw) - torch.log(1 - torch.tanh(raw).pow(2) + 1e-6)
            ).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # Actor loss is the negative of the expected return (advantage) weighted log probability of the actions, plus an entropy regularization term to encourage exploration. Critic loss is the mean squared error between the predicted state values and the computed returns.
            actor_loss = -(log_p * adv_f).mean() - self.ent_coef * entropy

            # Update actor network
            self.opt_actor.zero_grad()
            actor_loss.backward()
            # Gradient clipping for stability, preventing excessively large updates that can destabilize training.
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
            self.opt_actor.step()

            val_pred = self.critic(obs_f).squeeze(-1)
            critic_loss = nn.functional.mse_loss(val_pred, ret_f)

            # Update critic network
            self.opt_critic.zero_grad()
            critic_loss.backward()
            # Gradient clipping for stability, preventing excessively large updates that can destabilize training.
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
            self.opt_critic.step()

            # Log training metrics for visualization and checkpointing.
            self.history["actor_loss"].append(actor_loss.item())
            self.history["critic_loss"].append(critic_loss.item())
            self.history["entropy"].append(entropy.item())

            recent_rets = self.history["ep_returns"][-20:] or [0.0]
            recent_lens = self.history["ep_lengths"][-20:] or [0]
            mean_ret = np.mean(recent_rets)

            # save best checkpoint
            if mean_ret > best_ret:
                best_ret = mean_ret
                self.save("actor_critic_best.pth")

            # Update progress bar with current training metrics, including total environment steps, mean return over recent episodes,
            # best return achieved so far, average episode length, actor loss, critic loss, and policy entropy.
            pbar.set_postfix(
                {
                    "steps": f"{total_env_steps / 1e3:.1f}k",
                    "ret": f"{mean_ret:.2f}",
                    "best": f"{best_ret:.2f}",
                    "ep_len": f"{np.mean(recent_lens):.0f}",
                    "a_loss": f"{actor_loss.item():.3f}",
                    "c_loss": f"{critic_loss.item():.3f}",
                    "entropy": f"{entropy.item():.3f}",
                }
            )

        self.envs.close()
        return self.history

    def save(self, path: str = "actor_critic.pth"):
        # Save the actor and critic network weights to a checkpoint file. This allows for later loading and evaluation or resuming training.
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )
        print(f"Saved -> {path}")

    def load(self, path: str = "actor_critic.pth"):
        # Load the actor and critic network weights from a checkpoint file.
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"Loaded <- {path}")

    def plot_training(self, save_path: str = "training_curves.png"):
        # Plot training curves for episode returns, episode lengths, actor loss, critic loss, and policy entropy over the course of training.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("A2C Training — Unbalanced Disk", fontsize=14, fontweight="bold")

        def smooth(x, w=50):
            # Simple moving average smoothing for better visualization of trends in noisy training curves. If the input array is shorter than the window size, it returns the original array without smoothing.
            return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x

        ax = axes[0, 0]
        if self.history["ep_returns"]:
            ax.plot(self.history["ep_returns"], alpha=0.3, color="#4c8cbf")
            ax.plot(smooth(self.history["ep_returns"]), color="#4c8cbf", lw=2)
        ax.set_title("Episode Return")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        if self.history["ep_lengths"]:
            ax.plot(self.history["ep_lengths"], alpha=0.3, color="#e07b39")
            ax.plot(smooth(self.history["ep_lengths"]), color="#e07b39", lw=2)
        ax.set_title("Episode Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(self.history["actor_loss"], color="#5ba85e", lw=1.5)
        ax.set_title("Actor Loss")
        ax.set_xlabel("Update")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax2 = ax.twinx()
        ax.plot(
            self.history["critic_loss"], color="#c44e52", lw=1.5, label="Critic MSE"
        )
        ax2.plot(
            self.history["entropy"],
            color="#8172b2",
            lw=1.5,
            linestyle="--",
            label="Entropy",
        )
        ax.set_title("Critic Loss & Entropy")
        ax.set_xlabel("Update")
        ax.set_ylabel("Critic MSE", color="#c44e52")
        ax2.set_ylabel("Entropy", color="#8172b2")
        ax.grid(alpha=0.3)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves → {save_path}")


#########################################################################
#  DEMO — renders live with pygame + prints reward per step
#########################################################################


def demo(trainer: A2CTrainer, env_cls, env_kwargs: dict, n_steps: int = 500):
    """Greedy rollout with live rendering and reward printed to terminal."""
    import time

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
            theta_ref = getattr(env, "th_ref", 0.0)

            err = ((theta - theta_ref + np.pi) % (2 * np.pi)) - np.pi
            # err_norm = err / np.pi
            # aug      = torch.tensor([sin_th, cos_th, omega, err_norm],
            #                          dtype=torch.float32).unsqueeze(0)

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mu, _ = trainer.actor(obs_t)
            action = (3.0 * torch.tanh(mu)).item()

            obs, reward, term, trunc, _ = env.step(action)

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
            time.sleep(env.dt)

            if term or trunc:
                obs, _ = env.reset()

    env.close()
    trainer.actor.train()

    # Plots
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

    errs = np.array(
        [((th - r + np.pi) % (2 * np.pi) - np.pi) for th, r in zip(thetas, refs)]
    )
    print("\nDemo trajectory saved -> demo_trajectory.png")
    print(f"Mean reward : {np.mean(rewards):.4f}")
    print(f"RMSE        : {np.rad2deg(np.sqrt(np.mean(errs**2))):.2f}°")
    print(f"Max |V|     : {np.max(np.abs(voltages)):.3f} V")
    print(f"Mean |ω|    : {np.mean(np.abs(omegas)):.3f} rad/s")


#########################################################################
#  MAIN
#########################################################################

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    sys.path.insert(0, os.getcwd())
    from UnbalancedDisk_latest import UnbalancedDisk_sincos

    ENV_CLS = UnbalancedDisk_sincos
    ENV_KWARGS = dict(
        umax=3.0, dt=0.025, randomise=False
    )  # set randomise=True during training for robustness, False for final demo

    trainer = A2CTrainer(
        env_cls=ENV_CLS,
        env_kwargs=ENV_KWARGS,
        n_envs=8,  # parallel workers via multiprocessing
        n_steps=64,
        gamma=0.99,
        lam=0.95,
        lr_actor=3e-4,
        lr_critic=1e-3,
        ent_coef=0.005,  # higher entropy -> more exploration
        total_steps=500_000,  # 2M steps for swing-up to emerge
        hidden=256,
    )

    # trainer.train()
    # trainer.save("actor_critic_final.pth")                  # save final too
    trainer.load("actor_critic_best.pth")  # load best for demo
    trainer.plot_training("training_curves.png")
    ENV_KWARGS = dict(umax=3.0, dt=0.025, randomise=False)  # deterministic env for demo
    demo(trainer, ENV_CLS, ENV_KWARGS, n_steps=500)
