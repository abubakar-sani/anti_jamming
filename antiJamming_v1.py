#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
import tf_slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from tensorflow import keras
from ns3gym import ns3env

env = gym.make('ns3-v0')
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space, ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n
jammerType = 'combined'

model = keras.Sequential()
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(a_size, activation='softmax'))
model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

total_episodes = 1
max_env_steps = 1
env._max_episode_steps = max_env_steps

epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.99

time_history = []
rew_history = []

# Training agent
for e in range(total_episodes):

    state = env.reset()
    state = np.reshape(state, [1, s_size])
    rewardsum = 0
    for time in range(max_env_steps):
        # Choose action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(a_size)
        else:
            action = np.argmax(model.predict(state)[0])

        # Step
        next_state, reward, done, _ = env.step(action)

        if done or time == max_env_steps - 1:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, rewardsum, epsilon))
            break

        next_state = np.reshape(next_state, [1, s_size])

        # Train
        target = reward
        if not done:
            target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))

        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay

    time_history.append(time)
    rew_history.append(rewardsum)
    # Implementing early break

# Plotting Learning Performance
print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10, 4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")  # , color='red')
plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")  # , color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('learning.pdf', bbox_inches='tight')
plt.show()

# for n in range(2 ** s_size):
#    state = [n >> i & 1 for i in range(0, 2)]
#    state = np.reshape(state, [1, s_size])
#    print("state " + str(state)
#        + " -> prediction " + str(model.predict(state)[0])
#        )

# Testing agent
n_runs = 1
total_trans_pkts = 0

for run in range(n_runs):
    state = env.reset()
    state = np.reshape(state, [1, s_size])
    total_trans_pkts_per_run = 0
    for time in range(max_env_steps):
        # Choose Channel
        action = np.argmax(model.predict(state)[0])
        # Step
        next_state, reward, done, _ = env.step(action)
        total_trans_pkts_per_run += reward
        if done or time == max_env_steps - 1:
            break
        next_state = np.reshape(next_state, [1, s_size])
        # Test
        state = next_state

    print(f"Run: {run}/{n_runs}, Total transferred packets: {total_trans_pkts_per_run}")
    total_trans_pkts += total_trans_pkts_per_run

# print(model.get_config())
# print(model.to_json())
# print(model.get_weights())

# Save Results for this time slots value
normalizedThroughput = total_trans_pkts / (100 * n_runs)
print(f'The normalized throughput is: {normalizedThroughput}')
filename = f'{jammerType}_timeSlots_{max_env_steps}.json'
with open(filename, 'w') as f:
    json.dump(normalizedThroughput, f)
