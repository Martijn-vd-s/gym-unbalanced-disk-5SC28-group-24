# Actor-Critic -- Unbalanced Disk

Two RL algorithms are implemented: **PPO** (recommended) and **A2C**.
Both train a policy to swing up and balance the disk, with optional reference tracking.

---

## Training

### PPO
```bash
cd src/actor_critic
python ppo.py
```
Trains for 1.5M steps using 8 parallel environments with domain randomisation on.
Saves the best checkpoint to `ppo_best.pth` and training history to `ppo_training_history.npz`.

Key hyperparameters at the bottom of `ppo.py`:
| Parameter | Default | What it does |
|---|---|---|
| `n_envs` | 8 | Number of parallel workers -- more = faster but more RAM |
| `n_steps` | 256 | Steps collected per worker before each update |
| `ppo_epochs` | 10 | Gradient update passes over each batch |
| `lr_actor` | 1e-4 | Actor learning rate |
| `ent_coef` | 0.01 | Entropy bonus -- increase if policy converges too early |
| `total_steps` | 1_500_000 | Total training steps -- increase for harder tasks |

### A2C
```bash
cd src/actor_critic
python actor_critic_a2c.py
```
Trains for 1.5M steps using 8 parallel environments.
Saves the best checkpoint to `actor_critic_best.pth` and history to `a2c_training_history.npz`.

---

## Running a trained policy -- `runExp.py`

`runExp.py` loads a saved policy and runs a live demo episode with rendering and plots.

**Steps:**
1. Make sure your `.pth` checkpoint file is in `src/actor_critic/`
2. Edit the loader line near the top of `runExp.py`:
   ```python
   trainer.load("ppo_best_tracking.pth")  # change to your checkpoint name
   ```
3. If you want to test with a different environment, change `ENV_CLS`:
   ```python
   from UnbalancedDiskExp import UnbalancedDisk_exp_sincos  # real hardware wrapper
   # or:
   from UnbalancedDisk import UnbalancedDisk_sincos          # sim environment
   ```
4. Run from the `src/actor_critic/` folder:
   ```bash
   python runExp.py
   ```

The script prints a step-by-step table (angle, error, voltage, reward), renders the disk live, and saves two plots when done:
- `demo_trajectory.png` -- angle vs time
- `demo_phase_diagram.png` -- phase portrait (theta vs omega)

### Setting the reference angle manually

By default the environment drives `th_ref` automatically with a 0.2 Hz square wave (+/-15 deg around pi).
To set it yourself, turn off `_auto_ref` after the reset and then write `env.th_ref` directly inside the loop in `runExp.py`:

```python
obs, _ = env.reset()
env._auto_ref = False          # stop the automatic square-wave reference
env.th_ref = np.pi             # start at straight up (pi radians)

for step in range(n_steps):
    # change the reference whenever you want, e.g. every 100 steps
    if step == 100:
        env.th_ref = np.pi + np.deg2rad(15)   # 15 deg to the right
    if step == 200:
        env.th_ref = np.pi - np.deg2rad(15)   # 15 deg to the left

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action = trainer.actor(obs_t).item()
    obs, reward, term, trunc, _ = env.step(action)
    ...
```

`th_ref` is in radians. Straight up is `np.pi`. The training used steps of +/-15 degrees so staying within that range gives the best results.

---

## Reward function tuning -- `UnbalancedDisk.py`

The reward is defined in `UnbalancedDisk._reward()` and has four components:

```
reward = r_balance + r_swing + r_track - u_penalty - rate_penalty
```

### Components and what to tune

**`r_balance`** -- wide Gaussian that rewards being near upright
```python
sigma_err = np.pi / 4.0   # width ~45 deg; make smaller to demand more precision
```
This gives a gradient across the full swing-up range. Rarely needs changing.

**`r_swing`** -- S-curve swing-up shaping
```python
A = 12.0           # peak target velocity; raise if disk is too slow to swing up
sigma_swing = 2.0  # tolerance around target velocity
```
If the policy gets stuck spinning without gaining height, try increasing `A`.

**`r_track`** -- narrow Gaussian for fine tracking (weighted 1.5x)
```python
sigma_track = np.deg2rad(7.0)   # tighten to demand closer tracking (e.g. 5 deg)
r_track = 1.5 * np.exp(...)    # increase weight to prioritise tracking over swing-up
```
This term dominates near the top and drives the policy to close the reference error.
Increase the weight if the policy balances but does not follow step references.

**`u_penalty`** -- penalises large voltages
```python
u_penalty = 0.07 * u_norm**2 + 0.03 * abs(u_norm)
```
Raise the coefficients if the trained policy applies unnecessarily large voltages on the real hardware.

**`rate_penalty`** -- penalises rapid voltage changes
```python
rate_penalty = 0.10 * (u_norm - prev_u_norm)**2
```
Raise if the motor chatters or the voltage signal is noisy/oscillatory.

### General tips
- Train with `randomise=True` -- it randomises physical parameters and a motor deadzone each episode, which improves sim-to-real transfer.
- If swing-up works but tracking is poor: increase the `r_track` weight or tighten `sigma_track`.
- If the policy never learns to swing up: reduce penalties or increase `A` in `r_swing`.
- If the policy is aggressive on real hardware: increase `u_penalty` and `rate_penalty`.
