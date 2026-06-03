import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
from collections import deque


class UnbalancedDisk(gym.Env):
    """
    UnbalancedDisk
    th =
                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = starting location
    """

    def __init__(self, umax=3.0, dt=0.025, render_mode="human", randomise=False):
        ############# start do not edit  ################
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001

        # self.g = 9.80155078791343
        # self.J = 0.000244210523960356
        # self.Km = 10.5081817407479
        # self.I = 0.0410772235841364
        # self.M = 0.0761844495320390
        # self.tau = 0.397973147009910
        ############# end do not edit ###################
        self.randomise = randomise
        if randomise:  # <- enable during training, disable for real deployment
            self.omega0 *= np.random.uniform(0.9, 1.1)  # 10%
            self.gamma *= np.random.uniform(0.8, 1.2)  # 20%
            self.Ku *= np.random.uniform(0.9, 1.1)
            self.Fc *= np.random.uniform(0.8, 1.2)

        self.umax = umax
        self.dt = dt  # time step

        self.th_ref = np.pi  # reference angle for reward calculation
        self.th_before = 0
        self._th_accumulated = 0
        self.prev_th = 0

        # change anything here (compilable with the exercise instructions)
        self.action_space = spaces.Box(
            low=-umax, high=umax, shape=tuple()
        )  # continuous
        # self.action_space = spaces.Discrete(5) #discrete
        low = [-float("inf"), -40]
        high = [float("inf"), 40]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(2,),
        )

        # self.reward_fun = lambda self: np.exp(-(self.th%(2*np.pi)-np.pi)**2/(2*(np.pi/7)**2)) #example reward function, change this!

        self.err = lambda self: (
            ((self.th - self.th_ref + np.pi) % (2 * np.pi)) - np.pi
        )
        # self.reward_fun = lambda self: np.exp(-(self.err(self)**2) / (2 * 0.1**2))
        # W1 = 10
        # W_BOT_SPIN = 0.1
        # W_TOP = 100

        # self.reward_fun = lambda self: (
        #     W1 * np.cos(self.err(self))
        #     - min(W_BOT_SPIN * max(0, abs(self.th)-(4/3)*np.pi) ** 2, 300)
        #     + W_TOP * np.exp(-5 * (self.err(self))**2 - 0.02 * (self.omega - self.th_ref)**2)
        # )

        # self.reward_fun = lambda self: (
        #     (2 * np.cos(self.err(self))
        #     - 0.01 * self.omega**2                          # back to 0.01
        #     + 10 * np.exp(-0.1*self.err(self)**2)           # wide Gaussian (kept)
        #     - 10.0 * np.exp(-0.05*(self.err(self)**2 - np.pi**2)**2 / (2 * (np.pi/4)**2))
        #             * np.exp(-0.1*self.omega**2 / (2 * 0.5**2))
        #             * (1 - np.exp(-self.err(self)**2 / (2 * 0.3**2)))
        #     # - 0.005 * (self.u / umax)**2                     # ← ADD: small effort penalty
        #     # + 10 * np.exp(-2*self.err(self)**2 - 2*self.omega**2)
        #     ) / 12
        # )

        # worked
        # self.reward_fun = lambda self: (
        # (2 * np.cos(self.err(self))
        # - 0.01 * self.omega**2
        # + 10 * np.exp(-0.1*self.err(self)**2 - 0.0*self.omega**2)
        # - 10.0 * np.exp(-0.05*(self.err(self)**2 - np.pi**2)**2 / (2 * (np.pi/4)**2))  # peaks at err=±π (bottom)
        #         * np.exp(-0.1*self.omega**2 / (2 * 0.5**2))
        #         # * (1 - np.exp(-self.err(self)**2 / (2 * 0.3**2)))
        # # + 1 * np.exp(-self.err(self)**2-self.omega**2)
        # )/ 12
        # # - 50.0 * (abs(self.th) < np.pi / 2) * (abs(self.omega) < 0.5)
        # )

        # self.reward_fun = lambda self: (
        #     (2 * np.cos(self.err(self))                          # smooth -1→+1 guidance
        #     - 0.01 * self.omega**2                               # speed penalty
        #     + 10 * np.exp(-0.1 * self.err(self)**2)              # sharp top bonus, no omega inside
        #     - 5.0 * (1 - np.cos(self.err(self)))**2              # ← bottom penalty, EXACTLY 0 at err=0
        #             * np.exp(-self.omega**2 / (2 * 0.5**2))
        #     ) / 12
        # )

        # self.reward_fun = lambda self: self._reward()

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0  # for visual
        self.reset()

    def _reward(
        self,
        A=14.0,
        sigma_err=np.pi,  # much wider: allow big angles
        sigma_omega=3.0,
        w_ridge=10.0,
    ):  # stronger ridge
        err = self.err(self)
        omega = self.omega

        base = (
            -0.01 * omega**2
            + 10 * np.exp(-0.1 * err**2)
            - 10.0
            * np.exp(-0.05 * (err**2 - np.pi**2) ** 2 / (2 * (np.pi / 4) ** 2))
            * np.exp(-0.1 * omega**2 / (2 * 0.5**2))
            * (1 - np.exp(-(err**2) / (2 * 0.3**2)))
            + 2 * np.exp(-2 * err**2 - 2 * omega**2)
        )

        # ridge term (symmetric magnitude)
        omega_target = A * np.sin(err)
        angle_term = np.exp(-(err**2) / (2 * sigma_err**2))
        vel_term = np.exp(
            -((np.abs(omega) - np.abs(omega_target)) ** 2) / (2 * sigma_omega**2)
        )
        ridge = angle_term * vel_term

        reward = (base + w_ridge * ridge) / 12.0
        return reward

    def step(self, action):
        # convert action to u
        terminated = False
        self.u = action  # continuous
        # self.u = [-3,-1,0,1,3][action] #discrate
        # self.u = [-3,3][action] #discrate
        ##### Start Do not edit ######
        self.u = np.clip(self.u, -self.umax, self.umax)

        def f(t, y):
            th, omega = y
            dthdt = omega
            friction = self.gamma * omega + self.Fc * np.tanh(
                omega / self.coulomb_omega
            )
            domegadt = (
                -(self.omega0**2) * np.sin(th + self.delta_th)
                - friction
                + self.Ku * self.u
            )
            return np.array([dthdt, domegadt])

        sol = solve_ivp(f, [0, self.dt], [self.th, self.omega])  # integration
        self.th, self.omega = sol.y[:, -1]

        # ### ive edited this to be more efficient, but it should give the same result as the solve_ivp above
        # y        = np.array([self.th, self.omega])
        # friction = self.gamma*y[1] + self.Fc*np.tanh(y[1]/self.coulomb_omega)
        # dydt     = np.array([y[1], -self.omega0**2*np.sin(y[0]+self.delta_th) - friction + self.Ku*self.u])
        # self.th, self.omega = y + self.dt * dydt

        ##### End do not edit   #####

        # accumulate total rotation
        # self._th_accumulated += self.th - self.th_before
        # self.th_before = self.th
        ## terminate if spun more than 3π in total (reward hacking prevention)
        # spin_limit = 3 * np.pi
        # terminated = abs(self._th_accumulated) > spin_limit
        reward = self.reward_fun(self)

        # if terminated:
        #     reward -=  10  # heavy penalty for spinning too much

        return self.get_obs(), reward, terminated, False, {}

    def reset(self, seed=None):
        self.th = np.random.normal(loc=0, scale=0.001)
        self.omega = np.random.normal(loc=0, scale=0.001)
        self.u = 0
        self._th_accumulated = 0.0
        self.th_before = 0.0

        if self.randomise:
            self.omega0 = 11.339846957335382 * np.random.uniform(0.9, 1.1)
            self.gamma = 1.3328339309394384 * np.random.uniform(0.8, 1.2)
            self.Ku = 28.136158407237073 * np.random.uniform(0.9, 1.1)
            self.Fc = 6.062729509386865 * np.random.uniform(0.8, 1.2)

        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.001)  # do not edit
        self.omega_noise = self.omega + np.random.normal(
            loc=0, scale=0.001
        )  # do not edit
        return np.array([self.th_noise, self.omega_noise])

    def render(self):
        import pygame
        from pygame import gfxdraw

        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega  # x = self.state

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
        gfxdraw.filled_circle(  # small midle disk
            self.surf,
            screen_width // 2,
            screen_height // 2,
            int(screen_width / 2 * 0.06 * 1.3),
            (132, 132, 126),
        )

        from math import cos, sin

        r = screen_width // 2 * 0.40 * 1.3
        gfxdraw.filled_circle(  # disk
            self.surf,
            int(screen_width // 2 - sin(th) * r),  # is direction correct?
            int(screen_height // 2 - cos(th) * r),
            int(screen_width / 2 * 0.22 * 1.3),
            (155, 140, 108),
        )
        gfxdraw.filled_circle(  # small nut
            self.surf,
            int(screen_width // 2 - sin(th) * r),  # is direction correct?
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
                    raise ValueError(f"u={u} is not the correct shape")
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
            self.viewer.blit(
                arrow_rot,
                (
                    screen_width // 2 - arrow_size // 2,
                    screen_height // 2 - arrow_size // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        return True

    def close(self):
        if self.viewer is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.viewer = None


class UnbalancedDisk_sincos(UnbalancedDisk):
    """docstring for UnbalancedDisk_sincos"""

    def __init__(self, umax=3.0, dt=0.025, randomise=False):
        super(UnbalancedDisk_sincos, self).__init__(
            umax=umax, dt=dt, randomise=randomise
        )
        low = [-1, -1, -40.0, -40.0]
        high = [1, 1, 40.0, 40.0]
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(4,),
        )  ### change shape here!!!

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0, scale=0.005)  # do not edit
        self.omega_noise = self.omega + np.random.normal(
            loc=0, scale=0.005
        )  # do not edit

        err_noise = ((self.th_noise - self.th_ref + np.pi) % (2 * np.pi)) - np.pi

        # dth = (self.th_noise - self.prev_th) / self.dt  # finite diff velocity
        # self.prev_th = self.th_noise

        return np.array(
            [
                np.sin(self.th_noise),
                np.cos(self.th_noise),
                self.omega_noise,
                (err_noise / np.pi),
            ]
        )  # change anything here


if __name__ == "__main__":
    import time

    env = UnbalancedDisk(dt=0.025)

    obs = env.reset()
    Y = [obs]
    env.render()
    try:
        for i in range(100):
            time.sleep(1 / 24)
            u = 3  # env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(u)
            Y.append(obs)
            env.render()
    finally:
        env.close()
    from matplotlib import pyplot as plt
    import numpy as np

    Y = np.array(Y)
    plt.plot(Y[:, 0])
    plt.title(f"max(Y[:,0])={max(Y[:, 0])}")
    plt.show()
