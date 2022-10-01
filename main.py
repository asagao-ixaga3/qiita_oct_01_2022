
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory

from factory_for_casestudy import FactoryForCasestudy

config_debug = dict(
    debug_mode = True,
    n_eps = 3,
    sample_batch_size = 2**5, 
    num_steps = 2**1,
    num_train_iteration = 2**2)

config_main = dict(
    debug_mode = False,
    n_eps = 2**6,
    sample_batch_size = 2**5, 
    num_steps = 2**3,
    num_train_iteration = 2**6)

get_mean_abs_omega = lambda oscillator_train: np.sum(np.mean(np.abs(oscillator_train.Omega), axis=0)) 

config = config_debug

factory = FactoryForCasestudy()

train_env, _, oscillator_train = factory.createEnvironment(debug_mode = config["debug_mode"])

actor_net_a, critic_net_a = factory.createActorAndCriticNetworks(train_env)
train_agent_a, _ = factory.createAgent(train_env, actor_net_a, critic_net_a)

actor_net_b, critic_net_b = factory.createActorAndCriticNetworks(train_env)
train_agent_b, _ = factory.createAgent(train_env, actor_net_b, critic_net_b)

replay_buffer = factory.createReplayBuffer(train_agent_a)

log_reward = []

for i_eps in range(config["n_eps"]):
    
    id_eps = tf.constant((-1,), dtype=tf.int64)
    
    train_agent_b._actor_network.set_weights(train_agent_a._actor_network.get_weights()) 
            
    train_env.reset()
    while True:        
        time_step_for_agent_a = train_env.current_time_step()  
        action_step_of_agent_a = train_agent_a.collect_policy.action(time_step_for_agent_a) 
        time_step_for_agent_b = train_env.step(action_step_of_agent_a.action) 
        action_step_of_agent_b = train_agent_b.policy.action(time_step_for_agent_b) 
        next_time_step_for_agent_a = train_env.step(action_step_of_agent_b.action)         

        traj = trajectory.from_transition(time_step_for_agent_a, action_step_of_agent_a, next_time_step_for_agent_a) 
        id_eps = replay_buffer.add_batch(traj, id_eps)
        if next_time_step_for_agent_a.is_last():
            break                                
    
    log_reward.append((i_eps, get_mean_abs_omega(oscillator_train)))
    
    iterator = iter(replay_buffer.as_dataset(sample_batch_size = config["sample_batch_size"], num_steps = config["num_steps"]))
    for _ in range(config["num_train_iteration"]): 
        trajectories, _ = next(iterator)
        train_agent_a.train(experience=trajectories)
