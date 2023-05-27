#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import mkdir
import gym
import tensorflow as tf
import tf_slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from tensorflow import keras
from ns3gym import ns3env
from DDQN_CNN import DoubleDeepQNetwork
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


env = gym.make('ns3-v0')
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space, ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n
jammerType = 'hopping'
network = 'CNN'
csc = 0  # Channel switching cost

total_episodes = 200
max_env_steps = 100
train_end = 0
TRAIN_Episodes = 100
remaining_Episodes = 0
env._max_episode_steps = max_env_steps

epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999
gamma = 0.95
lr = 0.001
experience_replay_batch_size = 32
history = 1
batch_size = 1

DDQN_agent = DoubleDeepQNetwork(s_size, a_size, history, batch_size, lr, gamma, epsilon, epsilon_min, epsilon_decay)
rewards = []  # Store rewards for graphing
epsilons = []  # Store the Explore/Exploit

# Training agent
for e in range(TRAIN_Episodes):
    state = env.reset()
    state = np.array(state, dtype='float32')
    # print(f"Initial state is: {state}")
    state = np.reshape(state, [batch_size, 1, history, s_size])  # Resize to store in memory to pass to .predict
    tot_rewards = 0
    previous_action = 0
    for time in range(max_env_steps):  # 200 is when you "solve" the game. This can continue forever as far as I know
        action = DDQN_agent.action(state)
        next_state, reward, done, _ = env.step(action)
        # print(f'The next state is: {next_state}')
        # done: Three collisions occurred in the last 10 steps.
        # time == max_env_steps - 1 : No collisions occurred
        if done or time == max_env_steps - 1:
            rewards.append(tot_rewards)
            epsilons.append(DDQN_agent.epsilon)
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e, TRAIN_Episodes, tot_rewards, DDQN_agent.epsilon))
            break
        # Applying channel switching cost
        if action != previous_action:
            reward -= csc
        next_state = np.array(next_state, dtype='float32')
        next_state = np.reshape(next_state, [batch_size, 1, history, s_size])
        tot_rewards += reward
        DDQN_agent.store(state, action, reward, next_state, done)  # Resize to store in memory to pass to .predict
        state = next_state
        previous_action = action

        # Experience Replay
        if len(DDQN_agent.memory) > experience_replay_batch_size:
            DDQN_agent.experience_replay(experience_replay_batch_size)
    # Update the weights after each episode (You can configure this for x steps as well
    DDQN_agent.update_target_from_model()
    # If our current NN passes we are done
    # Early stopping criteria: I am going to use the last 10 runs within 1% of the max
    if len(rewards) > 10 and np.average(rewards[-10:]) >= max_env_steps - 0.05 * max_env_steps:
        # Set the rest of the episodes for testing
        remaining_Episodes = total_episodes - e
        train_end = e
        break

# Testing
print('Training complete. Testing started...')
# TEST Time
#   In this section we ALWAYS use exploit as we don't train anymore
total_transmissions = 0
successful_transmissions = 0
if remaining_Episodes == 0:
    train_end = TRAIN_Episodes
    TEST_Episodes = 100
else:
    TEST_Episodes = total_episodes - train_end
# Testing Loop
n_channel_switches = 0
for e_test in range(TEST_Episodes):
    state = env.reset()
    state = np.array(state, dtype='float32')
    # print(f"Initial state is: {state}")
    state = np.reshape(state, [batch_size, 1, history, s_size])
    tot_rewards = 0
    previous_channel = 0
    for t_test in range(max_env_steps):
        action = DDQN_agent.test_action(state)
        next_state, reward, done, _ = env.step(action)
        if done or t_test == max_env_steps - 1:
            rewards.append(tot_rewards)
            epsilons.append(0)  # We are doing full exploit
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e_test, TEST_Episodes, tot_rewards, 0))
            break
        next_state = np.array(next_state, dtype='float32')
        next_state = np.reshape(next_state, [batch_size, 1, history, s_size])
        tot_rewards += reward
        if action != previous_channel:
            n_channel_switches += 1
        if reward == 1:
            successful_transmissions += 1
        # DON'T STORE ANYTHING DURING TESTING
        state = next_state
        previous_channel = action
        # done: More than 3 collisions occurred in the last 10 steps.
        # t_test == max_env_steps - 1: No collisions occurred
        total_transmissions += 1

# Plotting
plotName = f'results/{network}/{jammerType}_csc_{csc}.png'
rolling_average = np.convolve(rewards, np.ones(10) / 10)
plt.plot(rewards)
plt.plot(rolling_average, color='black')
plt.axhline(y=max_env_steps - 0.10 * max_env_steps, color='r', linestyle='-')  # Solved Line
# Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range
eps_graph = [200 * x for x in epsilons]
plt.plot(eps_graph, color='g', linestyle='-')
# Plot the line where TESTING begins
plt.axvline(x=train_end, color='y', linestyle='-')
plt.xlim((0, train_end+TEST_Episodes))
plt.ylim((0, max_env_steps))
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig(plotName, bbox_inches='tight')
plt.show()

# Save Results
# Rewards
fileName = f'results/{network}/rewards_{jammerType}_csc_{csc}.json'
with open(fileName, 'w') as f:
    json.dump(rewards, f)
# Normalized throughput
normalizedThroughput = successful_transmissions / (TEST_Episodes*(max_env_steps-2))
print(f'The normalized throughput is: {normalizedThroughput}')
fileName = f'results/{network}/throughput_{jammerType}_csc_{csc}.json'
with open(fileName, 'w') as f:
    json.dump(normalizedThroughput, f)
# Channel switching times
normalized_cst = n_channel_switches / (TEST_Episodes*(max_env_steps-2))
print(f'The normalized channel switching times is: {normalized_cst}')
fileName = f'results/{network}/times_{jammerType}_csc_{csc}.json'
with open(fileName, 'w') as f:
    json.dump(normalized_cst, f)
# Save the agent as a SavedAgent.
agentName = f'savedAgents/{network}/DDQNAgent_{jammerType}_csc_{csc}'
DDQN_agent.save_model(agentName)
