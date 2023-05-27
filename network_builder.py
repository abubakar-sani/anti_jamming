#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import random

batch_size = 1
history = 1
channel = 1
state = [0, 0, 1, 0, 0]
# Line for implementing FIFO
# historical_state = np.zeros((history, s_size), dtype='type')
# historical_state[1:-1] = historical_state[:-2]; historical_state[0] = next_state
state = np.array(state, dtype='float32')
state = np.reshape(state, [batch_size, channel, history, state.shape[0]])
y = Conv2D(32, (1, 1), activation='relu', input_shape=state.shape)(state)
print('First y is:')
print(y)
y = Conv2D(32, (1, 1), activation='relu')(y)
print('The second y is:')
print(y)
y = Dropout(0.5)(y)
print('After Dropout')
print(y)
# y = MaxPooling2D(pool_size=2, strides=(1, 2), padding='valid')
y = Flatten()(y)
print('After flattening')
print(y)
y = Dense(10, activation='relu')(y)
print(y)
y = Dense(5, activation='softmax')(y)
print('Output is:')
print(y)

