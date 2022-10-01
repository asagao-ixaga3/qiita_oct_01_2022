import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.environments import tf_py_environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.network import Network
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.train.utils import train_utils

from oscillator import Simulator
from environments import Environment


class FactoryForCasestudy(object):
    
    def createAgent(self, train_env: TFPyEnvironment, actor_net: Network, critic_net: Network, learning_rate = 3e-2) -> TFAgent:
        
        critic_learning_rate = learning_rate # @param {type:"number"}
        actor_learning_rate = learning_rate # @param {type:"number"}
        alpha_learning_rate = learning_rate # @param {type:"number"}
        target_update_tau = 0.005 # @param {type:"number"}
        target_update_period = 1 # @param {type:"number"}
        gamma = 1.0 # @param {type:"number"}
        reward_scale_factor = 1.0 # @param {type:"number"}
        
        action_spec, time_step_spec = train_env.action_spec(), train_env.time_step_spec()
        
        train_step = train_utils.create_train_step()
        
        tf_agent = sac_agent.SacAgent(
              time_step_spec,
              action_spec,
              actor_network=actor_net,
              critic_network=critic_net,
              actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                  learning_rate=actor_learning_rate),
              critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                  learning_rate=critic_learning_rate),
              alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                  learning_rate=alpha_learning_rate),
              target_update_tau=target_update_tau,
              target_update_period=target_update_period,
              td_errors_loss_fn=tf.math.squared_difference,
              gamma=gamma,
              reward_scale_factor=reward_scale_factor,
              train_step_counter=train_step)
        
        tf_agent.initialize()
        
        return tf_agent, train_step

    def createReplayBuffer(self, train_agent: TFAgent, replay_buffer_capacity = 2**16) -> ReplayBuffer:
    
        replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
            data_spec = train_agent.collect_data_spec,
            capacity = replay_buffer_capacity,
            completed_only = True)
            
        return replay_buffer
    
    def createActorAndCriticNetworks(self, train_env, network_type = "rnn", **args):
        
        if network_type == "fnn":
            return self._createActorAndCriticNetworksFnn(train_env, **args)

        if network_type == "rnn":
            return self._createActorAndCriticNetworksRnn(train_env, **args)
        
        assert False

    def _createActorAndCriticNetworksRnn(self, train_env, actor_fc_layer_params = (32,), critic_joint_fc_layer_params = (32,), lstm_size = (2**3,), **args):
        
        observation_spec, action_spec, _ = train_env.observation_spec(), train_env.action_spec(), train_env.time_step_spec()
        
        actor_net = ActorDistributionRnnNetwork(
              observation_spec,
              action_spec,
              lstm_size = lstm_size,
              input_fc_layer_params = actor_fc_layer_params,
              output_fc_layer_params = actor_fc_layer_params,
              continuous_projection_net=(
                  tanh_normal_projection_network.TanhNormalProjectionNetwork))

        critic_net = CriticRnnNetwork(
                (observation_spec, action_spec),
                lstm_size = lstm_size,
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')
        
        return actor_net, critic_net
    
    def _createActorAndCriticNetworksFnn(self, train_env, actor_fc_layer_params = (), critic_joint_fc_layer_params = (32,), **args):

        observation_spec, action_spec = train_env.observation_spec(), train_env.action_spec()
    
        actor_net = actor_distribution_network.ActorDistributionNetwork(
              observation_spec,
              action_spec,
              fc_layer_params=actor_fc_layer_params,
              continuous_projection_net=(
                  tanh_normal_projection_network.TanhNormalProjectionNetwork))

        critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')

        return actor_net, critic_net
    
    def createEnvironment(self, debug_mode, discount = 0.9, control_interval = 2*np.pi/2**3, dt = 2*np.pi/2**6):
        
        if debug_mode:
            simulation_period = 2*np.pi
        else:
            simulation_period = 2*np.pi * 2**4

        oscillator_train = Simulator(dt, simulation_period)

        train_env = Environment(discount = discount, osc = oscillator_train, control_interval = control_interval)        
        
        return tf_py_environment.TFPyEnvironment(train_env), train_env, oscillator_train