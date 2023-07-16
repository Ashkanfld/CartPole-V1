# import needed packages
import time
from collections import deque, namedtuple
import h5py
import gymnasium as gym
import PIL.Image
import numpy as np
import tensorflow as tf
import utils
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

tf.random.set_seed('1234')

# Hyperparameters 
SEED = 0              		# seed for pseudo-random number generator
MEMORY_SIZE = 1_000_000		# size of memory buffer
GAMMA = 0.995			# discount factor
ALPHA = 1e-4			# learning rate
NUM_STEPS_FOR_UPDATE = 4	# perform a learning update every C time steps
MINIBATCH_SIZE = 64		# mini-batch size		
E_DECAY = 0.995			# ε decay rate for ε-greedy policy
E_MIN = 0.01			# minimum ε value for ε-greedy policy		
TAU = 1e-3            		# soft update parameter
random.seed(SEED)

# loading the environment from gymnasium
env = gym.make('CartPole-v1')

# Action and Observation space
state_size = env.observation_space.shape
num_action = env.action_space.n

# reset the Environment and get the initial state
initial_state, _ = env.reset()

# create the Q-Network
q_network = Sequential ([
    Input(shape = state_size),
    Dense(units = 64, activation = 'relu', name = 'layer1'),
    Dense(units = 64, activation = 'relu', name = 'layer2'),
    Dense(units = num_action, activation = 'linear' , name = 'layer3'),
], name = 'q_network')

target_q_network = Sequential ([
    Input(shape = state_size),
    Dense(units = 64, activation = 'relu', name = 'layer1'),
    Dense(units = 64, activation = 'relu', name = 'layer2'),
    Dense(units = num_action, activation = 'linear' , name = 'layer3'),
], name = 'target_q_network')
optimizer = Adam(learning_rate = ALPHA)

# Store experiences as named tuples
experience = namedtuple('Experience', field_names = ['state', 'action', 'reward' , 'next_state', 'done'])

# calculate Mean squared error loss 
def compute_loss(experiences, q_network, target_q_network, gamma):
    
    states, actions, rewards, next_states, done_vals = experiences
    max_qsa = tf.reduce_max(target_q_network(next_states),axis = -1)
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),tf.cast(actions, tf.int32)],axis=1))
    loss = MSE(y_targets, q_values)
    
    return loss

# update the Network weights 
@tf.function
def agent_learn(experiences, gamma):
    
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, q_network, target_q_network, gamma)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    
    update_target_network(q_network, target_q_network)

# get an action 
def get_action(q_network, epsilon):
    if random.random() > epsilon:
        return np.argmax(q_network.numpy()[0])
    else:
        return random.choice(np.arange(num_action))

# check whether any update is needed 
def check_update_conditions(t, num_steps, memory_buffer):
    if (t + 1) % num_steps == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True 
    else:
        return False 
        
# updating epsilon 
def get_new_eps(epsilon):
    return max(epsilon*E_DECAY, E_MIN)

# update the target Q-network 
def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

# extracting values from experience 
def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)


# Train the agent 

start = time.time()

num_episodes = 10000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100
epsilon = 1.0

memory_buffer = deque(maxlen = MEMORY_SIZE)

target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    
    state, _ = env.reset()
    total_points = 0 
    
    for t in range(0,max_num_timesteps):
        
        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        action = get_action(q_values, epsilon) 
        next_state, reward, done, truncated,_= env.step(action)
        
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            
            experiences = get_experiences(memory_buffer)
                        
            agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        
        total_points += reward 
        
        if done or truncated:
            
            break
    
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    epsilon = get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
    # We will consider that the environment is solved if we get an
    # average of 450 points in the last 100 episodes.
    if av_latest_points >= 500 * 0.9:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('Cart_Pole_final_result.h5')
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
  













