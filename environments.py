
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import numpy as np
from oscillator import Simulator


class Environment(py_environment.PyEnvironment):
        
    def __init__(self, n_action = 1, discount=0.9, osc:Simulator=None, control_interval=None):
        '''
        Observations consist of 5 variables:
            (omega_a, omega_b, omega_load, p_load, p_source_b) for agent a,
            (omega_b, omega_a, omega_load, p_load, p_source_a) for agent b.
        
        An action of agent a(agent b) determines the value of p_source_a(p_source_b).
        '''        
        super(Environment, self).__init__()
        
        self.n_observation = 5        
        self.n_action = n_action

        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.n_observation,), dtype=np.float32, minimum=(-1,), maximum=(1,), name='observation')        
        self._action_spec = array_spec.BoundedArraySpec(shape=(self.n_action,), dtype=np.float32, minimum=(-1,), maximum=(1,), name='action')

        self._episode_ended = False
        
        self.discount = discount        
        self.osc = osc
        self.control_interval_step = int(control_interval/osc._dt)
        
        self.mode = None
        self.p_source_a = None
        self.p_source_b = None
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def get_reward(self):
        omega_a, omega_b, omega_load = self.osc.get_omega()
        cost = np.abs(omega_a)/(2*np.pi) + np.abs(omega_b)/(2*np.pi) + np.abs(omega_load)/(2*np.pi)
        reward = -cost
        return reward

    def get_observation_for_agent_a(self):
        
        omega_a, omega_b, omega_load = self.osc.get_omega()    
        p_load = self.osc.get_p_load()
        p_source_a, p_source_b = self.osc.get_p_source()
            
        return np.clip([omega_a/(2*np.pi), omega_b/(2*np.pi), omega_load/(2*np.pi), p_load, p_source_b], -1, 1).astype(np.float32) # (2*n_self.oscillator+3,)

    def get_observation_for_agent_b(self):

        omega_a, omega_b, omega_load = self.osc.get_omega()    
        p_load = self.osc.get_p_load()
        p_source_a, p_source_b = self.osc.get_p_source()
            
        return np.clip([omega_b/(2*np.pi), omega_a/(2*np.pi), omega_load/(2*np.pi), p_load, p_source_a], -1, 1).astype(np.float32) # (2*n_self.oscillator+3,)
    
    def _reset(self):
        
        self.osc.reset()
        self._episode_ended = False
        self.mode = 0
        
        return ts.restart(observation = self.get_observation_for_agent_a())

    def _step(self, action_from_an_agent):
        # action_from_an_agent: (1,)
        
        if self._episode_ended:
            return self.reset()
        
        if self.mode == 0:
            '''
            When mode = 0,
                interpret the value of action as p_source_a, 
                return observations for agent b, 
                and wait until the next action has arrived. 
            '''            
            self.p_source_a = action_from_an_agent[0]
            self.mode = 1
            
            reward_dummy = 0.0
            observation = self.get_observation_for_agent_b()
            return ts.transition(observation = observation, reward = reward_dummy, discount = self.discount)
        
        if self.mode == 1:
            '''
            When mode = 1,                 
                interpret the value of action as p_source_b,
                develop the simulation by several steps,  
                and return observations for agent a, 
            '''
            self.p_source_b = action_from_an_agent[0]
            self.mode = 0      
            
            rewards = []
            for _ in range(self.control_interval_step):
                done = self.osc.step(p_source = [self.p_source_a, self.p_source_b])
                rewards.append(self.get_reward())
                if done:
                    break
            
            reward = np.mean(rewards)
            observation = self.get_observation_for_agent_a()
            
            if done:
                self._episode_ended = True
                return ts.termination(observation = observation, reward = reward)
            else:
                self._episode_ended = False
                return ts.transition(observation = observation, reward = reward, discount = self.discount)