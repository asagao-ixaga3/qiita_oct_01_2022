
import numpy as np

class Simulator(object):
    
    def __init__(self, dt, simulation_period, p_load_random_state = np.random.RandomState(), p_load_amp = 1.0, p_load_update_cycle = 2*np.pi):

        self._n_oscillator = 3
        self._b = 1.0
        self._dt = dt
        self._cnt = None
        self._t = None
        self._x = None        
        self._nsteps = int(simulation_period/dt)
        self._n_agent = 2
        
        self.p_load = None
        self.p_load_update_cycle = p_load_update_cycle
        self.p_load_amp = p_load_amp
        self.p_load_random_state = p_load_random_state
        
    def reset(self):        
        
        self._cnt = 0
        self._t = 0.
        self._x = np.zeros(2*self._n_oscillator) 
        self.p_load = 0.0
        
        self.Theta = np.zeros((self._nsteps+1, self._n_oscillator)) 
        self.Omega = np.zeros((self._nsteps+1, self._n_oscillator)) 
        self.Psource = np.zeros((self._nsteps, self._n_agent)) 
        self.Pload = np.zeros(self._nsteps)         
        self.Time =  np.zeros(self._nsteps+1) 
                
        self.Time[self._cnt] = self._t
        self.Theta[self._cnt,:] = self._x[:self._n_oscillator]
        self.Omega[self._cnt,:] = self._x[self._n_oscillator:2*self._n_oscillator]
        
    def step(self, p_source):
        '''
        p_source consists of p_source_a and p_source_b.
        '''
        
        self.Psource[self._cnt, :] = p_source
        self.Pload[self._cnt] = self.p_load
        
        k0 = self.f(self._x, p_load = self.p_load, p_source = p_source)
        k1 = self.f(self._x + k0 * self._dt/2, p_load = self.p_load, p_source = p_source)
        k2 = self.f(self._x + k1 * self._dt/2, p_load = self.p_load, p_source = p_source)
        k3 = self.f(self._x + k2 * self._dt, p_load = self.p_load, p_source = p_source)

        self._cnt += 1        
        self._t += self._dt    
        self._x += (k0+2*k1+2*k2+k3)/6 * self._dt # (2*n_oscillator+1,)
        self.update_p_load()

        self.Time[self._cnt] = self._t
        self.Theta[self._cnt,:] = self._x[:self._n_oscillator]
        self.Omega[self._cnt,:] = self._x[self._n_oscillator:2*self._n_oscillator]
        
        done = self._cnt == self._nsteps
        
        return done

    def update_p_load(self):

        if self.p_load_random_state.rand() < 1./(self.p_load_update_cycle/self._dt):
            self.p_load = self.p_load_amp * (2*self.p_load_random_state.rand()-1)
        
    def f(self, x, p_source, p_load):
        '''
        x has (6,)-shape.
        * x[:3] indicates theta_a, theta_b, and theta_load
        * x[3:6] indicates omega_a, omega_b, and omega_load
        
        p_source consists of p_source_a and p_source_b.
        '''
    
        n_oscillator = self._n_oscillator
        b = self._b
        
        theta_a, theta_b, theta_n = x[:n_oscillator]
        
        p_source_a, p_source_b = p_source
        omega = x[n_oscillator:2*n_oscillator]
        
        dxdt = np.zeros(2*n_oscillator)
        
        dxdt[:n_oscillator] = omega 
    
        p_an = b * np.sin(theta_a - theta_n) # power flow from a to load
        p_bn = b * np.sin(theta_b - theta_n) # power flow from b to load
        p_ab = b * np.sin(theta_a - theta_b) # power flow from a to b
        p_ba = -p_ab # power flow from b to a

        dxdt[n_oscillator] = p_source_a - p_an - p_ab
        dxdt[n_oscillator+1] = p_source_b - p_bn - p_ba
        dxdt[n_oscillator+2] = -p_load + p_bn + p_an
                
        return dxdt
    
    def get_p_source(self):
        if self._cnt > 0:
            return self.Psource[self._cnt-1,:].copy() 
        else:
            return (0., 0.)

    def get_p_load(self):
        if self._cnt > 0:
            return self.Pload[self._cnt-1].copy() 
        else:
            return 0.
    
    def get_omega(self):
        return self.Omega[self._cnt,:].copy() 
    