"""
Unbalanced-disk environment for tabular Q-learning (simulation).

This is the Gymnasium environment used to *train* the swing-up + balance policy
in simulation. It exposes a discrete action set (a list of motor voltages) and a
2-D continuous observation ``[theta, omega]`` that is discretised by the
``Discretize_obs`` wrapper in the training notebook.

Conventions (angle ``theta``)::

                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = hanging down (start)

So ``theta = pi`` (or ``-pi``) is the upright/top position, ``theta = 0`` is the
bottom. The physical parameters below come from our own system identification of
the real setup; the deployment environment (``UnbalancedDiskExp``) and the
runExp script must use the same action map / observation range to stay
consistent with a trained Q-table.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path


class UnbalancedDisk(gym.Env):
    """Discrete-action unbalanced disk for swing-up + balance.

    The agent picks one of ``num_actions`` discrete voltages each step. The
    continuous dynamics are integrated with ``solve_ivp`` over one sample time
    ``dt``. The reward is shaped to (a) strongly reward the upright position
    with low angular velocity, (b) penalise hanging at the bottom, and (c) keep
    control gentle and smooth *only* near the top (so it does not get pushed off
    on the real hardware) without hindering the energy-pumping swing-up.
    """

    def __init__(self, umax=3.0, dt=0.025, render_mode='human'):
        # --- Physical parameters (from our own system identification) ---
        self.omega0 = 12.7908          # natural frequency term
        self.delta_th = 0              # angle offset
        self.gamma = 2.1904            # viscous friction coefficient
        self.Ku = 30.4070              # voltage-to-torque gain
        self.Fc = 9.1626               # Coulomb friction magnitude
        self.coulomb_omega = 0.001     # smoothing speed for the tanh friction model

        self.umax = umax               # max |voltage|
        self.dt = dt                   # sample time [s]
        self.render_mode = render_mode

        # --- Discrete action set: motor voltages [V] ---
        # Fine spacing near zero for gentle holding at the top, coarse for
        # strong swing-up pumping. NB: this list MUST match UnbalancedDiskExp.py
        # and runExp.py, or a trained Q-table maps indices to the wrong voltages.
        self.num_actions = 12
        self.action_space = spaces.Discrete(self.num_actions)
        self.discrete_action_map = [-3, -2.2, -1.5, -0.9, -0.5, -0.2, 0.2, 0.5, 0.9, 1.5, 2.2, 3]
        # self.discrete_action_map = [-3, -1.5, -0.5, 0.5, 1.5, 3] #RBF test

        # --- Observation: [theta, omega], clipped to this range by the wrapper ---
        low = [-(5 / 4) * np.pi, -5]
        high = [(5 / 4) * np.pi, 5]
        self.observation_space = spaces.Box(low=np.array(low, dtype=np.float32),
                                            high=np.array(high, dtype=np.float32), shape=(2,))

        # Wrapped absolute angular error between current angle and a target angle.
        self.err = lambda current_th, target_th: abs(((current_th - target_th + np.pi) % (2 * np.pi)) - np.pi)

        def gaussian_2d(val_x, val_y, mu_x, mu_y, sigma_x, sigma_y, rho, scale):
            """Scaled 2-D Gaussian of (val_x, val_y) about (mu_x, mu_y)."""
            mu = np.array([mu_x, mu_y])
            cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                            [rho * sigma_x * sigma_y, sigma_y**2]])
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            diff = np.array([val_x - mu[0], val_y - mu[1]])
            exponent = -0.5 * np.dot(diff, np.dot(inv_cov, diff.T))
            return scale * (1.0 / (2 * np.pi * np.sqrt(det_cov))) * np.exp(exponent)

        # --- Parameters of the extra "z" reward surface (calculate_z_component) ---
        self.A_VALUE = 3.25
        self.B_VALUE = 0.7
        self.J_VALUE = 2.0

        def _calculate_z_internal():
            """Extra hand-crafted reward surface; contributes 0 outside its mask."""
            term1 = np.sin(self.omega - np.sin(self.th) * self.A_VALUE + 0.5 * np.pi)
            term2 = (self.J_VALUE + np.sin(self.th - 0.5 * np.pi) * 2)
            z = term1 * self.B_VALUE * term2

            condition_y_lower = np.sin(self.th) * self.A_VALUE - np.pi
            condition_y_upper = np.sin(self.th) * self.A_VALUE + np.pi
            condition_z_lower = -self.J_VALUE + 2

            mask = (self.omega >= condition_y_lower) & \
                   (self.omega <= condition_y_upper) & \
                   (z >= condition_z_lower)
            return np.where(mask, z, 0)

        self.calculate_z_component = _calculate_z_internal

        # --- Sim-to-real shaping: gentle, smooth control only near the top ---
        self.TOP_GATE_SIGMA = 0.25     # rad: width of the "near top" zone (wider -> gentleness starts earlier)
        self.W_U_TOP = 0.34            # weight: penalise voltage^2 near top AND nearly stopped
        self.HOLD_OMEGA_SIGMA = 2.0    # rad/s: |omega| below this counts as "holding"
        self.W_RATE_TOP = 0.01         # weight: penalise rapid voltage switching near top
        # ~1 near the top, ~0 elsewhere -> shaping only acts where fine balancing is needed.
        self.top_gate = lambda: np.exp(-0.55 * (self.err(self.th, np.pi) / self.TOP_GATE_SIGMA) ** 2)
        # ~1 when nearly stopped (holding), ~0 at high speed (catching/swing-up) -> never blocks the catch.
        self.hold_gate = lambda: np.exp(-0.45 * (self.omega / self.HOLD_OMEGA_SIGMA) ** 2)
        # extra shaping:
        self.U_BOUND = 2.0                                  # near the very top, voltage above this is penalised
        self.W_BOUND = 0.7                                  # weight of that bound penalty
        self.TOP5_SIGMA = np.deg2rad(8)                     # ~+-5 deg 'very near top' zone
        self.top5_gate = lambda: np.exp(-0.6 * (self.err(self.th, np.pi) / self.TOP5_SIGMA) ** 2)
        self.W_SPEED = 0.09                                 # weight of the approach-speed reward
        self.SPEED_BAND = (np.deg2rad(1), np.deg2rad(18))   # err-to-top band where carrying speed is good
        self.OMEGA_CAP = 12.0                               # cap for the speed reward

        # --- Reward function ---
        self.reward_fun = lambda self_instance: (
            # main peak: upright (theta=pi) with low angular velocity
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 1, 1, 0.0, 2)
            # penalty for hanging at the bottom (theta ~ 0)
            - gaussian_2d(self_instance.err(self_instance.th, 0), self_instance.omega, 0, 0, 3, 3, 0.0, 40)
            # two sharper bonuses to favour the exact top
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 0.15, 0.15, 0.0, 0.3)
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 0.07, 0.07, 0.0, 0.05)
            # voltage-magnitude penalty: only near the top AND nearly stopped (holding)
            - self_instance.W_U_TOP * self_instance.top_gate() * self_instance.hold_gate() * self_instance.u**2
            # action-rate penalty (anti-chatter): only near the top, leaves swing-up pumping free
            - self_instance.W_RATE_TOP * self_instance.top_gate() * (self_instance.u - self_instance.prev_u)**2
            # near the very top (+-5 deg) AND nearly stopped: penalise only voltage above U_BOUND (~1.5 V)
            - self_instance.W_BOUND * self_instance.top5_gate() * self_instance.hold_gate() * np.maximum(0.0, np.abs(self_instance.u) - self_instance.U_BOUND) ** 2
            # reward carrying speed on the approach (~5..40 deg from the top) so it reaches the top with energy
            + self_instance.W_SPEED * ((self_instance.err(self_instance.th, np.pi) > self_instance.SPEED_BAND[0]) & (self_instance.err(self_instance.th, np.pi) < self_instance.SPEED_BAND[1])) * np.clip(np.abs(self_instance.omega), 0.0, self_instance.OMEGA_CAP)
            # extra reward surface
            + self_instance.calculate_z_component()
        )

        # --- Episode / rendering state ---
        self.viewer = None
        self.u = 0                     # current applied voltage (also used by render)
        self.prev_u = 0                # previous applied voltage (for the rate penalty)
        self.set_th = None             # if set, reset() forces this start angle (used in evaluation)
        self.set_omega = None          # if set, reset() forces this start velocity
        self.start_scale = 1.0         # curriculum knob: 1.0 = wide starts -> ~0 = bottom at rest (set by Qlearn)

        self.reset()

    def termination(self, reward):
        """Episode ends (with a penalty) if the disk has spun past +-2*pi."""
        if abs(self.th) > 2 * np.pi:
            done = True
            reward -= 50
        else:
            done = False
        return done, reward

    def step(self, action):
        """Apply the chosen discrete voltage, integrate one dt, return (obs, reward, done, truncated, info)."""
        self.prev_u = self.u                          # remember last voltage for the rate penalty
        self.u = self.discrete_action_map[action]

        ##### Start Do not edit ######
        self.u = np.clip(self.u, -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(omega / self.coulomb_omega)
            domegadt = -self.omega0**2 * np.sin(th + self.delta_th) - friction + self.Ku * self.u
            return np.array([dthdt, domegadt])

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])  # integration
        self.th, self.omega = sol.y[:, -1]
        ##### End do not edit   #####

        reward = self.reward_fun(self)
        done, reward = self.termination(reward)
        return self.get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        """Reset to a random start; ``start_scale`` shrinks the start range over training.

        With ``start_scale = 1`` the start is wide (anywhere up to ~+-144 deg,
        |omega| up to 2.8); as it shrinks toward 0 the start concentrates at the
        bottom at rest (the real deployment scenario). ``set_th`` / ``set_omega``
        override this with a fixed start (used by the evaluation cell).
        """
        super().reset(seed=seed)
        s = self.start_scale
        self.th = self.set_th if self.set_th is not None else np.random.uniform(-np.pi * (4 / 5) * s, np.pi * (4 / 5) * s)
        self.omega = self.set_omega if self.set_omega is not None else np.random.uniform(-2.8 * s, 2.8 * s)
        self.u = 0
        self.prev_u = 0
        return self.get_obs(), {}

    def get_obs(self):
        """Return the noisy observation ``[theta, omega]`` (small sensor noise, do not edit)."""
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)        # do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=0.001)  # do not edit
        return np.array([self.th_noise, self.omega_noise])

    def render(self):
        """Render the disk and the current voltage arrow with pygame."""
        import pygame
        from pygame import gfxdraw

        screen_width = 500
        screen_height = 500

        th = self.th

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        gfxdraw.filled_circle(  # central blue disk
            self.surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.65 * 1.3),
            (32, 60, 92),
        )
        gfxdraw.filled_circle(  # small middle disk
            self.surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.06 * 1.3),
            (132, 132, 126),
        )

        from math import cos, sin
        r = screen_width // 2 * 0.40 * 1.3
        gfxdraw.filled_circle(  # off-centre mass
            self.surf,
            int(screen_width // 2 - sin(th) * r),
            int(screen_height // 2 - cos(th) * r),
            int(screen_width / 2 * 0.22 * 1.3),
            (155, 140, 108),
        )
        gfxdraw.filled_circle(  # small nut
            self.surf,
            int(screen_width // 2 - sin(th) * r),
            int(screen_height // 2 - cos(th) * r),
            int(screen_width / 2 * 0.22 / 8 * 1.3),
            (71, 63, 48),
        )

        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray, list)):
                if self.u.ndim == 1:
                    u = self.u[0]
                elif self.u.ndim == 0:
                    u = self.u
                else:
                    raise ValueError(f'u={u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u) / self.umax * screen_height) * 0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow, Z)
            if self.u < 0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width // 2 - arrow_size // 2, screen_height // 2 - arrow_size // 2))
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        return True

    def close(self):
        """Close the pygame viewer if open."""
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.viewer = None


class UnbalancedDisk_sincos(UnbalancedDisk):
    """Variant whose observation is ``[sin(theta), cos(theta), omega]`` (for NN policies)."""

    def __init__(self, umax=3.0, dt=0.025):
        super(UnbalancedDisk_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1, -1, -40.]
        high = [1, 1, 40.]
        self.observation_space = spaces.Box(low=np.array(low, dtype=np.float32),
                                            high=np.array(high, dtype=np.float32), shape=(3,))

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)       # do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0, scale=1.0)  # do not edit
        return np.array([np.sin(self.th_noise), np.cos(self.th_noise), self.omega_noise])


if __name__ == '__main__':
    import time
    env = UnbalancedDisk(dt=0.025)

    obs = env.reset()
    Y = [obs]
    env.render()
    try:
        for i in range(100):
            time.sleep(1 / 24)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            Y.append(obs)
            env.render()
    finally:
        env.close()
    from matplotlib import pyplot as plt
    Y = np.array(Y)
    plt.plot(Y[:, 0])
    plt.title(f'max(Y[:,0])={max(Y[:,0])}')
    plt.show()
