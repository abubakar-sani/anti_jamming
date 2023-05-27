#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: Abubakar Sani Ali
# GNU Radio version: 3.8.1.0

###################################################################################
# Importing Libraries
###################################################################################
import json
import matplotlib.pyplot as plt
import numpy as np

# Combined Jammer
jammers = ['combined', 'sweeping', 'hopping', 'complete']
network = 'FNN'
history = 1
cscs = [0, 0.1, 0.2, 0.3, 0.4]
max_env_steps = 100
Episodes = 200
for jammer in jammers:
    rewards = []
    for csc in range(len(cscs)):
        filename = f'./results/{network}/M={history}/rewards_{jammer}_csc_{cscs[csc]}.json'
        with open(filename) as file:
            content = json.load(file)
            content = np.convolve(content, np.ones(10) / 10)
            rewards.append(content)

    # experiment 1 plot: Rewards
    plotName = f'results/{network}/rewards_{jammer}.png'
    # rolling_average = np.convolve(rewards, np.ones(10) / 10)
    plt.plot(rewards[0], label='CSC = 0', color='red', linewidth=1.5,
             marker='o', markerfacecolor='red', markersize=0)
    plt.plot(rewards[1], label='CSC = 0.1', color='orange', linewidth=1.5,
             marker='o', markerfacecolor='orange', markersize=0)
    plt.plot(rewards[2], label='CSC = 0.2', color='green', linewidth=1.5,
             marker='o', markerfacecolor='green', markersize=0)
    plt.plot(rewards[3], label='CSC = 0.3', color='brown', linewidth=1.5,
             marker='o', markerfacecolor='brown', markersize=0)
    plt.plot(rewards[4], label='CSC = 0.4', color='blue', linewidth=1.5,
             marker='o', markerfacecolor='blue', markersize=0)

    plt.axhline(y=max_env_steps - 0.10 * max_env_steps, color='black', linestyle='-')  # Solved Line
    # Plot the line where TESTING begins
    plt.xlim((0, Episodes))
    plt.ylim((0, max_env_steps))
    plt.xlabel('Episodes')
    plt.ylabel('Rolling average rewards')
    plt.legend()
    plt.savefig(plotName, bbox_inches='tight')
    plt.show()

    # experiment 2 plots: Throughput
    throughput = []
    for csc in range(len(cscs)):
        filename = f'./results/{network}/M={history}/throughput_{jammer}_csc_{cscs[csc]}.json'
        with open(filename) as file:
            content = json.load(file)
            throughput.append(content)
    plotName = f'results/{network}/throughput_{jammer}.png'
    plt.bar(cscs[0], throughput[0], width=0.05, color='red')
    plt.bar(cscs[1], throughput[1], width=0.05, color='orange')
    plt.bar(cscs[2], throughput[2], width=0.05, color='green')
    plt.bar(cscs[3], throughput[3], width=0.05, color='brown')
    plt.bar(cscs[4], throughput[4], width=0.05, color='blue')

    plt.ylim((0, 1))
    plt.xlabel('Channel switching cost (CSC)')
    plt.ylabel('Normalized Throughput')
    plt.savefig(plotName, bbox_inches='tight')
    plt.show()

    # experiment 3 plots: Channel switching times
    cstime = []
    for csc in range(len(cscs)):
        filename = f'./results/{network}/M={history}/times_{jammer}_csc_{cscs[csc]}.json'
        with open(filename) as file:
            content = json.load(file)
            cstime.append(content)
    plotName = f'results/{network}/times_{jammer}.png'
    plt.bar(cscs[0], cstime[0], width=0.05, color='red')
    plt.bar(cscs[1], cstime[1], width=0.05, color='orange')
    plt.bar(cscs[2], cstime[2], width=0.05, color='green')
    plt.bar(cscs[3], cstime[3], width=0.05, color='brown')
    plt.bar(cscs[4], cstime[4], width=0.05, color='blue')

    # Plot the line where TESTING begins
    plt.ylim((0, 1))
    plt.xlabel('Channel switching cost (CSC)')
    plt.ylabel('Normalized channel switching frequency')
    plt.savefig(plotName, bbox_inches='tight')
    plt.show()
