# -*- coding: utf-8 -*-
"""DoubleDQN - CartPole.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1epgMUiYUSqgOMKWl4MQt3IHjNFVrfH7m
"""

import random
import gym
import numpy as np
import tensorflow
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
print(state_size)
print(env.observation_space)

action_size = env.action_space.n
print(action_size)
print(env.action_space)

batch_size = 50
n_episodes = 130
output_dir = 'model_output/cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Double Q-Learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_freq = 50
        self.updates = 0
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
                  
    def replay(self, batch_size):
        minibatch = random.sample(list(self.memory), batch_size)
        states = np.empty((batch_size, 4))
        actions = np.empty(batch_size)
        rewards = np.empty(batch_size)
        next_states = np.empty((batch_size, 4))
        dones = np.empty(batch_size)
        i = 0
        for state, action, reward, next_state, done in minibatch:
          states[i] = state
          actions[i] = action
          rewards[i] = reward
          next_states[i] = next_state
          dones[i] = done
          i += 1

        targets_s = self.model.predict(states)
        dones = np.reshape(dones, (-1, 1))
        rewards = np.reshape(rewards, (-1, 1))
        targets = self.target_model.predict(next_states)
        args_max = np.argmax(self.model.predict(next_states), axis=1)
        targets = np.asarray([[targets[i][args_max[i]] for j in range(targets.shape[1])] for i in range(targets.shape[0])])
        targets = rewards + (1-dones)*self.gamma*(targets)
        actions_one_hot = to_categorical(actions, self.action_size)
        actions_one_hot_neg = np.array(actions_one_hot == 0, dtype=np.uint8)
        targets = actions_one_hot_neg * targets_s + actions_one_hot * targets

        self.updates += 1
        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.updates % self.update_target_freq == 0:
          self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                  
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)    
done = False
scores = []
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    score = 0
    while not done:
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, score, agent.epsilon))
            
        if len(agent.memory) > batch_size:
          agent.replay(batch_size)

    scores.append(score)

plt.title('Double Q-Learning')
plt.plot(scores[:130])
plt.ylabel("score")
plt.xlabel("episodes")