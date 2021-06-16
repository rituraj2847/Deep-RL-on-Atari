#Training an agent to play Atari Game - Breakout using DQN

import random
import gym
import numpy as np
import cv2
import os
import sys
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

print(tensorflow.config.list_physical_devices("GPU"))


env = gym.make('BreakoutDeterministic-v4')
print(env.observation_space)

action_size = env.action_space.n
print(env.action_space)
print(env.unwrapped.get_action_meanings())

output_dir = 'model_output/breakout'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class ReplayBuffer:
    def __init__(self, size=500000, input_shape=(84, 84), history_length=4, batch_size=32):
        self.size = size
        self.history_length = history_length
        self.input_shape = input_shape
        self.actions = np.empty(size, dtype=np.uint8)
        self.rewards = np.empty(size, dtype=np.float32)
        self.frames = np.empty((size, input_shape[0], input_shape[1]), dtype=np.uint8)
        self.done = np.empty(size, dtype=np.bool)
        self.batch_size = batch_size

        self.curr_idx = 0
        self.count = 0
        
        self.states = np.empty((self.batch_size, self.history_length, self.input_shape[0], self.input_shape[1]), dtype=np.float32)
        self.next_states = np.empty((self.batch_size, self.history_length, self.input_shape[0], self.input_shape[1]), dtype=np.float32)
    
    #add the transition to replay buffer
    def add_experience(self, action, frame, reward, done):
        self.actions[self.curr_idx] = action
        self.rewards[self.curr_idx] = reward
        self.done[self.curr_idx] = done
        self.frames[self.curr_idx, ...] = frame
        
        self.curr_idx = (self.curr_idx + 1) % self.size
        self.count = max(self.count, self.curr_idx)
        
    #return a minibatch of transitions from the replay buffer   
    def get_minibatch(self):
        indices = []
        for i in range(self.batch_size):
            while True:
                index = np.random.randint(self.history_length, self.count)
                if index >= self.curr_idx and index-self.history_length < self.curr_idx:
                    continue
                if self.done[index-self.history_length:index].any():
                    continue
                break
            indices.append(index)
        
        for i in range(self.batch_size):
            index = indices[i]
            self.states[i, ...] = (self.frames[index-self.history_length:index, ...])/255
            self.next_states[i, ...] = (self.frames[index-self.history_length+1:index+1, ...])/255

        states1 = np.transpose(self.states, axes=(0,2,3,1))
        next_states1 = np.transpose(self.next_states, axes=(0,2,3,1))
        
        return states1, self.actions[indices], self.rewards[indices], next_states1, self.done[indices]


class DQNAgent:
    def __init__(self, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.1, learning_rate=0.00025, load_weights=False, filename=""):
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.update_target_freq = 10_000
        self.model = self._build_model()
        self.target_model = self._build_model()
       
        if load_weights is True:
            self.load(filename)
    
    #initialize the deep neural network
    def _build_model(self):
        initializer = keras.initializers.VarianceScaling(scale=2)
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(84, 84, 4), kernel_initializer=initializer))
        model.add(Conv2D(64, (4, 4), strides=2, activation='relu', kernel_initializer=initializer))
        model.add(Conv2D(64, (4, 4), strides=1, activation='relu', kernel_initializer=initializer))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=keras.losses.Huber(delta=1.0), optimizer=Adam(lr=self.learning_rate))
        return model
    
    #update weights of target network from the online network every C steps
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    
    #preprocess the frame by cropping the relevant portion and converting it to Gray 
    def preprocess(self, state):
        shape = (84, 84)
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        frame = frame[34:34+160, :160]  # crop image
        frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
        
        return frame
    
    #keep rewards between -1 and 1
    def clip_reward(self, reward):
        if reward > 0:
            return 1
        if reward < 0:
            return -1
        else:
            return 0
    
    #preprocess the frame and add the transition to replay buffer
    def remember(self, action, state, reward, done):
        frame = self.preprocess(state)
        self.memory.add_experience(action, frame, reward, done)
        
    #returns the action given the current state - based on epsilon greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(np.reshape(state, (-1,84,84,4)))[0]
        return np.argmax(act_values)
                  
    #sample a minibatch of transitions from replay buffer and train the network
    def replay(self):
        states, actions, rewards, next_states, done = self.memory.get_minibatch()
        
        Q_values = self.model.predict(states)
        actions_onehot = to_categorical(actions, self.action_size)
        actions_onehot_neg = np.array(actions_onehot == 0, dtype=np.uint8)
        targets_next = np.repeat(np.reshape(rewards, (-1, 1)), self.action_size, axis=1)  + self.gamma * \
        np.reshape(1-done, (-1, 1)) * self.target_model.predict(next_states)

        targets_next = np.asarray([[np.amax(targets_next[j]) for i in range(targets_next.shape[1])] for j in range(targets_next.shape[0])])
        Q_values = targets_next * actions_onehot + Q_values * actions_onehot_neg

        self.model.fit(states, Q_values, epochs = 1, verbose = 0)
        
        #annealing epsilon linearly for first million frames
        if self.epsilon > self.epsilon_min:
            epsilon_decrement = (1 - self.epsilon_min)/10_000_00
            self.epsilon -= epsilon_decrement
                  
    #loading the model from file
    def load(self, filename):
        self.model.load_weights(filename)
        self.target_model.load_weights(filename)
    
    #saving the model to file
    def save(self, filename):
        self.model.save_weights(filename)


env = gym.make('BreakoutDeterministic-v4')
agent = DQNAgent(action_size=4)
episodes = 0
total_frames_to_train_for = 100_000_00
frames_trained_for = 0
batch_size = 32
max_steps_in_episode = 10_000
scores = []

while frames_trained_for < total_frames_to_train_for:
    episodes += 1
    env.reset()
    score = 0
    next_state, reward, done, info = env.step(1)                #start the game by FIRE action

    #initially stack the same frame 4 times
    state_stack = np.repeat(np.reshape(agent.preprocess(next_state)/255.0, (84,84,1)), 4, axis=2)
    terminal_life_lost = 0
    lives_left = 5

    for time in range(max_steps_in_episode):
        #env.render()
        action = 1 if terminal_life_lost else agent.get_action(state_stack)           #when a life is lost, FIRE, otherwise get action based on epsilon policy
        next_state, reward, done, info = env.step(action)
        reward = agent.clip_reward(reward)
        terminal_life_lost = 0

        if info['ale.lives'] < lives_left:
            terminal_life_lost = 1
            lives_left = info['ale.lives']

        agent.remember(action, next_state, reward, terminal_life_lost)

        #append the next_state to state_stack 
        state_stack = np.append(state_stack[:, :, 1:], np.reshape(agent.preprocess(next_state)/255.0, (84,84,1)), axis=2)
        score += reward
        
        #writeout the score and other info to file when the episode is done
        if done:
            sys.stdout = open("progress.txt", "a")
            print("episode: {}, score: {}, e: {:.2}, frames: {}/{}".format(episodes, score, agent.epsilon, frames_trained_for, total_frames_to_train_for))
            sys.stdout.close()
            break
        
        if(agent.memory.count > 50000):
            agent.replay()
         
        frames_trained_for += 1
        
        if frames_trained_for % agent.update_target_freq == 0:
            agent.update_target_network()
    
    #save the weights of network every 200 episodes
    if episodes % 200 == 0:
        agent.save(output_dir+"weights_"+'{0:4d}'.format(episodes) + ".hdf5")
    



