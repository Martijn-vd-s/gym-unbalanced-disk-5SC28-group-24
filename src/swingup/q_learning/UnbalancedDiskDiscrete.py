
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path

class UnbalancedDisk(gym.Env):
    '''
    UnbalancedDisk
    th =            
                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = starting location
    '''
    def __init__(self, umax=3., dt = 0.025, render_mode='human'):
        # # ############# start do not edit  ################
        # self.omega0 = 11.339846957335382
        # self.delta_th = 0
        # self.gamma = 1.3328339309394384
        # self.Ku = 28.136158407237073
        # self.Fc = 6.062729509386865
        # self.coulomb_omega = 0.001

        # # self.g = 9.80155078791343
        # # self.J = 0.000244210523960356
        # # self.Km = 10.5081817407479
        # # self.I = 0.0410772235841364
        # # self.M = 0.0761844495320390
        # # self.tau = 0.397973147009910
        # ############# end do not edit ###################
        self.omega0 = 12.7908
        self.delta_th = 0
        self.gamma = 2.1904
        self.Ku = 30.4070
        self.Fc = 9.1626
        self.coulomb_omega = 0.001



        self.umax = umax
        self.dt = dt #time step
        self.num_actions = 12
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.num_actions)
        # Fijne resolutie rond 0 voor zacht vasthouden bij de top (vult het gat tussen 0 en 0.5).
        # holdhoek per koppel (nieuwe params): 0.15->1.6deg, 0.35->3.7deg, 0.7->7.5deg, 1.8->20deg, 3->34deg
        # self.discrete_action_map  = [-3, -1.8, -0.7, -0.35, -0.15, 0, 0.15, 0.35, 0.7, 1.8, 3] #fine
        # self.discrete_action_map  = [-3, -1.8,  -0.7 ,   0.7, 1.8,  3] #1
        # self.discrete_action_map  = [-3, -1.2 ,  0, 1.2,  3] #1
        self.discrete_action_map  = [-3,  -2.2, -1.5,-0.9,  -0.5, -0.2,  0.2,0.5, 0.9, 1.5, 2.2, 3] #2
        # self.discrete_action_map  = [-3,  -1.7, -0.7,  -0.2, 0,  0.2, 0.7, 1.7, 3] #3
        low = [-(5/4)*np.pi,-5] 
        high = [(5/4)*np.pi,5]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))



        # AANGEPAST: self.err functie accepteert nu 'current_th' en 'target_th'
        self.err = lambda current_th, target_th: abs(((current_th - target_th + np.pi) % (2 * np.pi)) - np.pi)
        # self.err = lambda self: abs(((self.th - np.pi + np.pi) % (2 * np.pi)) - np.pi)
        # Helper function for 2D Gaussian
        def gaussian_2d(val_x, val_y, mu_x, mu_y, sigma_x, sigma_y, rho, scale):
            mu = np.array([mu_x, mu_y])
            cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                            [rho * sigma_x * sigma_y, sigma_y**2]])
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            diff = np.array([val_x - mu[0], val_y - mu[1]])
            exponent = -0.5 * np.dot(diff, np.dot(inv_cov, diff.T))
            return scale * (1.0 / (2 * np.pi * np.sqrt(det_cov))) * np.exp(exponent)

        # Nieuwe klasse attributen voor de parameters van calculate_z
        self.A_VALUE = 3.25
        self.B_VALUE = 0.7
        self.J_VALUE = 2.0 # Gebruik 2.0 zoals in de reward_function_for_plot

        def _calculate_z_internal():
            """
            Berekent de waarde van z op basis van de gegeven wiskundige formule,
            gebruikmakend van de huidige th, omega en constante waarden van de klasse.
            Ongepaste waarden worden op 0 gezet.
            """
            # De hoofdberekening voor z
            term1 = np.sin(self.omega - np.sin(self.th) * self.A_VALUE + 0.5 * np.pi)
            term2 = (self.J_VALUE + np.sin(self.th - 0.5 * np.pi) * 2)
            z = term1 * self.B_VALUE * term2

            # Toepassen van de voorwaarden
            # {sin(x) * a - pi <= y <= sin((x)) * a + pi}
            condition_y_lower = np.sin(self.th) * self.A_VALUE - np.pi
            condition_y_upper = np.sin(self.th) * self.A_VALUE + np.pi
            
            # {-j + 2 <= z}
            condition_z_lower = -self.J_VALUE + 2

            # Maak een masker voor de punten die niet aan de voorwaarden voldoen
            # Als een punt niet aan de voorwaarden voldoet, wordt de waarde ervan 0
            mask = (self.omega >= condition_y_lower) & \
                   (self.omega <= condition_y_upper) & \
                   (z >= condition_z_lower)
            
            # Geef 0 terug voor de punten die niet aan de voorwaarden voldoen
            return np.where(mask, z, 0)
        


        # Wijs de geneste functie toe aan een self variabele zodat reward_fun deze kan aanroepen.
        self.calculate_z_component = _calculate_z_internal



        # --- Sim-to-real shaping: zacht en glad sturen vlak bij de top ---
        self.TOP_GATE_SIGMA = 0.25  # rad (~26 deg): breedte van de "bij de top" zone (breder -> eerder zacht)
        self.W_U_TOP = 0.34         # straf op u^2 bij de top EN bijna stilstaand (hoger -> zachter vasthouden)
        self.HOLD_OMEGA_SIGMA = 2.0 # rad/s: "bijna stilstaand"; sneller telt als vangen -> geen straf
        self.W_RATE_TOP = 0.01      # straf op snelle koppelwisselingen (anti-chatter) bij de top
        # gate ~1 bij de top, ~0 elders -> shaping alleen waar fijn balanceren nodig is
        self.top_gate = lambda: np.exp(-0.6 * (self.err(self.th, np.pi) / self.TOP_GATE_SIGMA) ** 2)
        # hold_gate ~1 als bijna stilstaand (vasthouden), ~0 bij hoge snelheid (vangen/swing-up) -> blokkeert de vangst niet
        self.hold_gate = lambda: np.exp(-0.6 * (self.omega / self.HOLD_OMEGA_SIGMA) ** 2)

        # DE NIEUWE REWARD FUNCTIE
        self.reward_fun = lambda self_instance: (
            # Hoofdbeloning: Piek op PI (bovenkant) en 0 hoeksnelheid



            +gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega,  0, 0, 1, 1, 0.0, 2)
            # Straf voor zijn aan de onderkant (rond 0 rad), ongeacht th_ref
            - gaussian_2d(self_instance.err(self_instance.th, 0), self_instance.omega,  0, 0, 3, 3, 0.0, 40)
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 0.15, 0.15, 0.0, 0.2)
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 0.07, 0.07, 0.0, 0.024)


            # Voltage-magnitude penalty: alleen bij de top EN bijna stilstaand (vasthouden) -> niet tijdens vangen/swing-up
            - self_instance.W_U_TOP * self_instance.top_gate() * self_instance.hold_gate() * self_instance.u**2
            # Action-rate penalty (anti-chatter), alleen bij de top zodat pompen tijdens swing-up vrij blijft
            - self_instance.W_RATE_TOP * self_instance.top_gate() * (self_instance.u - self_instance.prev_u)**2

            # TOEVOEGING van de getransformeerde Z_VALUES
            # Aangepaste aanroep: self_instance.calculate_z_component() in plaats van self_instance.err.calculate_z_component()
            + self_instance.calculate_z_component()   # UIT: vage off-top reward
        )

        self.render_mode = render_mode
        self.viewer = None
        self.u = 0 #for visual
        self.prev_u = 0
        self.prev_th = 0
        self.stuck = 0
        self.err_upright = 0
        self.up = False
        self.balancing_ticker = 0 
        self.punishment_over_time = 0
        self.bonus_region = 0
        self.set_th = None
        self.set_omega = None
        self.start_scale = 1.0  # curriculum: 1.0=wijde start -> ~0=onderaan in rust (Qlearn zet dit)
        self.step_t = 0
        self.time_to_become_stable = 0
        self.th_ref = 0
        self.th_set = 0
        self.reset()

    def termination(self, reward):
        ## termination function for if the system has fallen
        if abs(self.th) > 2 * np.pi:
            # self.up = True
            done = True
            reward -= 50
            # print(f'system is up: {self.up}')

        # if self.up == True and self.err > 1.6:
        #     # reward -= 100
        #     self.balancing_reward = 0
        #     done = False
        #     # print("terminating due to fall!!!")
        else:
            done = False

        return done, reward

    def step(self, action):
        self.prev_u = self.u  # vorige toegepaste koppel -> voor action-rate penalty
        self.u = self.discrete_action_map[action]

        ##### Start Do not edit ######
        self.u = np.clip(self.u,-self.umax,self.umax)
        def f(t,y):
            th, omega = y
            dthdt = omega
            friction = self.gamma*omega + self.Fc*np.tanh(omega/self.coulomb_omega)
            domegadt = -self.omega0**2*np.sin(th+self.delta_th) - friction + self.Ku*self.u
            return np.array([dthdt, domegadt])
        sol = solve_ivp(f,[0,self.dt],[self.th,self.omega]) #integration
        self.th, self.omega = sol.y[:,-1]
        ##### End do not edit   #####

        # first let the system stabilize, then more the setpoint +- 15 deg.
        if self.time_to_become_stable >= 100:
            t = self.dt * self.step_t
            self.th_ref = np.deg2rad(self.th_set) * np.sin(2 * np.pi * 0.2 * t) # move set-point at 0.2 Hz +- 15 deg
            self.step_t += 1
        else:
            self.th_ref = 0
        # self.th_ref = 0
        self.time_to_become_stable += 1
        #####################################################################

        # t = self.dt * self.step_t
        # self.th_ref = np.deg2rad(15) * np.sin(2 * np.pi * 0.2 * t) # move set-point at 0.2 Hz +- 15 deg
        # self.step_t += 1

        reward = self.reward_fun(self)

        reward += self.bonus_region
        done, reward = self.termination(reward)

        return self.get_obs(), reward, done, False, {}
         
    def reset(self,seed=None):
        super().reset(seed=seed)
        s = self.start_scale  # 1.0=wijd, ->0=onderaan in rust
        self.th = self.set_th if self.set_th is not None else np.random.uniform(-np.pi*(4/5)*s, np.pi*(4/5)*s)
        self.omega = self.set_omega if self.set_omega is not None else np.random.uniform(-2.8*s, 2.8*s)
        self.u = 0
        self.prev_u = 0

        self.step_t = 0
        self.balancing_ticker = 0
        self.time_to_become_stable = 0
        self.bonus_region = 0
        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return np.array([self.th_noise, self.omega_noise])

    def render(self):
        import pygame
        from pygame import gfxdraw
        
        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega #x = self.state

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        gfxdraw.filled_circle( #central blue disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.65*1.3),
            (32,60,92),
        )
        gfxdraw.filled_circle( #small midle disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.06*1.3),
            (132,132,126),
        )
        
        from math import cos, sin
        r = screen_width//2*0.40*1.3
        gfxdraw.filled_circle( #disk
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22*1.3),
            (155,140,108),
        )
        gfxdraw.filled_circle( #small nut
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22/8*1.3),
            (71,63,48),
        )
        
        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray,list)):
                if self.u.ndim==1:
                    u = self.u[0]
                elif self.u.ndim==0:
                    u = self.u
                else:
                    raise ValueError(f'u={u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u)/self.umax*screen_height)*0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow,Z)
            if self.u<0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)
                
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width//2-arrow_size//2, screen_height//2-arrow_size//2))
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
    def __init__(self, umax=3., dt = 0.025):
        super(UnbalancedDisk_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        # self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.01) #do not edit
        return np.array([np.sin(self.th_noise), np.cos(self.th_noise), self.omega_noise]) #change anything here

if __name__ == '__main__':
    import time
    env = UnbalancedDisk(dt=0.025)

    obs = env.reset()
    Y = [obs]
    env.render()
    try:
        for i in range(100):
            time.sleep(1/24)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            Y.append(obs)
            env.render()
    finally:
        env.close()
    from matplotlib import pyplot as plt
    import numpy as np
    Y = np.array(Y)
    plt.plot(Y[:,0])
    plt.title(f'max(Y[:,0])={max(Y[:,0])}')
    plt.show()
    