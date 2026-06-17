import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import time
import usb.util

global dev, dev_active
dev_active = False

# todo:
# update documentation on install and usage

class UnbalancedDisk_exp(gym.Env):
    '''
    UnbalancedDisk_exp
    th =            
                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = starting location

    '''
    def __init__(self, umax=3., dt=0.025, force_restart_dev=False, inactivity_release_time=3, render_mode='human'):
        '''
        umax : the maximal allowable input
        dt : the sample time
        force_restart_dev : set to true to reset connection
        inactivity_release_time : If the setup has not recived any inputs for ~inactivity_release_time/20 seconds than the input will be set to zero automaticly
        '''
        global dev, dev_active
        if dev_active:
            self.dev = dev
        if not dev_active or force_restart_dev:
            self.init_dev()

        assert isinstance(inactivity_release_time, int)
        self.set_inactivity_release_time(inactivity_release_time)

        self.umax = umax
        self.dt = dt

        ### Gym things
        self.num_actions = 9
        self.action_space = spaces.Discrete(self.num_actions)
        # self.discrete_action_map  = [-3, -1.8,  -0.5 ,  0,  0.5, 1.8,  3]
        # self.discrete_action_map  = [-3,  -2, -1,  -0.5 , -0.2, 0,  0.2, 0.5, 1, 2, 3] 
        self.discrete_action_map  = [-3,  -2, -1,  -0.6 , 0,  0.6, 1, 2, 3]
        
        low = [-2*np.pi, -5] 
        high = [2*np.pi, 5]
        self.observation_space = spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), shape=(2,))

        # Custom error function
        self.err = lambda current_th, target_th: abs(((current_th - target_th + np.pi) % (2 * np.pi)) - np.pi)
        
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

        # Variables for calculate_z
        self.A_VALUE = 3.25
        self.B_VALUE = 0.7
        self.J_VALUE = 2.0 

        def _calculate_z_internal():
            """
            Berekent de waarde van z op basis van de gegeven wiskundige formule,
            gebruikmakend van de huidige th, omega en constante waarden van de klasse.
            Ongepaste waarden worden op 0 gezet.
            """
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

        # NEW REWARD FUNCTION
        self.reward_fun = lambda self_instance: (
            # Hoofdbeloning: Piek op PI (bovenkant) en 0 hoeksnelheid
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega,  0, 0, 1, 1, 0.0, 2)
            # Straf voor zijn aan de onderkant (rond 0 rad), ongeacht th_ref
            - gaussian_2d(self_instance.err(self_instance.th, 0), self_instance.omega,  0, 0, 3, 3, 0.0, 40)
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 0.15, 0.15, 0.0, 0.05)
            + gaussian_2d(self_instance.err(self_instance.th, np.pi), self_instance.omega, 0, 0, 0.07, 0.07, 0.0, 0.15)
            
            # Control input penalty
            - 0.001 * self_instance.u**2
            
            # TOEVOEGING van de getransformeerde Z_VALUES
            + self_instance.calculate_z_component()
        )

        # Viewer and logic variables
        self.render_mode = render_mode
        self.viewer = None
        self.u = 0 
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
        self.step_t = 0
        self.time_to_become_stable = 0
        self.th_ref = 0
        self.th_set = 0

    def init_encoder(self):
        data_w=[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)
        data_w=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)

    def set_inactivity_release_time(self, inactivity_release_time):
        self.inactivity_release_time=inactivity_release_time
        data_w=[1,1,0,0,0,0,0,self.inactivity_release_time,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)
        data_w=[0,1,0,0,0,0,0,self.inactivity_release_time,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)

    def init_dev(self):
        global dev, dev_active
        try: #try closing the engine
            usb.util.dispose_resources(dev)
        except NameError:
            pass
        dev = usb.core.find(idVendor=0x04b4, idProduct=0x8612)
        dev.set_configuration() #this throws and error if python cannot connect to the disk.
        self.dev = dev
        dev_active = True

    def termination(self, reward):
        ## termination function for if the system has fallen
        if abs(self.th) > 2 * np.pi:
            done = True
            reward -= 50
        else:
            done = False
        return done, reward

    def step(self, action):
        # convert discrete action to u
        self.u = self.discrete_action_map[action]
        # self.u = 0

        ##### Hardware interfacing ######
        self.u = np.clip(self.u, -self.umax, self.umax)

        DacMin, DacMax, Relais= -10, 10, 1
        digital_input = int((self.u-DacMin)/(DacMax-DacMin)*65536)
        digital_in_sec = divmod(digital_input,256)

        data_pack=[0,0,digital_in_sec[0],0,0,Relais,digital_in_sec[1],self.inactivity_release_time,0,0,0,0,0,0,0,0]
        self.dev.write(0x02,data_pack,10)
        
        start_t = time.time() #a more accurate waiter than time.sleep
        while time.time() - start_t < self.dt:
            pass
        
        obs = self.get_obs() # This also updates self.th and self.omega
        
        # Stabilization and set-point movement
        if self.time_to_become_stable >= 100:
            t = self.dt * self.step_t
            self.th_ref = np.deg2rad(self.th_set) * np.sin(2 * np.pi * 0.2 * t) # move set-point at 0.2 Hz +- 15 deg
            self.step_t += 1
        else:
            self.th_ref = 0
        self.time_to_become_stable += 1

        reward = self.reward_fun(self)
        reward += self.bonus_region
        done, reward = self.termination(reward)

        return obs, reward, done, False, {}
        
    def reset(self, seed=None):
        if seed is not None:
             super().reset(seed=seed)

        # Hardware-specific physical reset sequence
        theta_now = self.get_obs()[0]
        t_start = time.time()
        while time.time()-t_start<30:
            time.sleep(0.1)
            theta_new = self.get_obs()[0]
            if abs(theta_new-theta_now)==0:
                break
            theta_now = theta_new
        time.sleep(0.1)
        self.init_encoder()

        # Reset custom logic variables
        self.u = 0
        self.step_t = 0
        self.balancing_ticker = 0
        self.time_to_become_stable = 0
        self.bonus_region = 0

        return self.get_obs(), {}

    def get_obs(self):
        couldnotreadcounter = 0
        while True:
            try:
                self.data_pack_read=self.dev.read(0x86,16,1)
                break
            except usb.USBError as e:
                print('USB read error')
                couldnotreadcounter += 1
                time.sleep(0.001)
                if couldnotreadcounter>20:
                    raise e
        data = self.data_pack_read
        if data[4]<128:
            position=2*np.pi*(data[4]*65536+data[3]*256+data[2])/2000
        else:
            position=2*np.pi*(data[4]*65536+data[3]*256+data[2]-16777216)/2000
        
        d = data
        omega = d[10]*-3.644127510645671 + d[14]*2.01877019753875 + d[12]*1.6121463023483062 + d[9]*-0.013751126061226403 

        self.th = position
        self.omega = omega
        return np.array([self.th, self.omega])

    def render(self):
        import pygame
        from pygame import gfxdraw
        
        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega 

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        gfxdraw.filled_circle( 
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.65*1.3),
            (32,60,92),
        )
        gfxdraw.filled_circle( 
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.06*1.3),
            (132,132,126),
        )
        
        from math import cos, sin
        r = screen_width//2*0.40*1.3
        gfxdraw.filled_circle( 
            self.surf,
            int(screen_width//2-sin(th)*r), 
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22*1.3),
            (155,140,108),
        )
        gfxdraw.filled_circle( 
            self.surf,
            int(screen_width//2-sin(th)*r), 
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

    def close_viewer(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.viewer = None

    def close(self):
        global dev, dev_active
        if dev_active:
            usb.util.dispose_resources(self.dev)
            dev_active = False
        self.close_viewer()


class UnbalancedDisk_exp_sincos(UnbalancedDisk_exp):
    """docstring for UnbalancedDisk_exp_sincos"""
    def __init__(self, umax=3., dt=0.025):
        super(UnbalancedDisk_exp_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32), high=np.array(high,dtype=np.float32), shape=(3,))

    def get_obs(self):
        super(UnbalancedDisk_exp_sincos, self).get_obs()
        return np.array([
            np.sin(self.th), 
            np.cos(self.th), 
            (self.omega + 1.874),
            # (err / np.pi),
            # (self.u / self.umax)
        ])

if __name__ == '__main__':
    # Add a simple physical test sequence
    env = UnbalancedDisk_exp(dt=0.025)

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
    Y = np.array(Y)
    plt.plot(Y[:,0])
    plt.title(f'max(Y[:,0])={max(Y[:,0])}')
    plt.show()