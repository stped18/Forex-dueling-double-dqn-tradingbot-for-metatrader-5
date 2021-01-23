import pickle
import os.path
import io
from os import path
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
import numpy as np
from tensorflow.python.keras.layers import Concatenate


class modelStore(object):
    def __init__(self,q_vale, q_next):
        self.q_vale=q_vale
        self.q_next=q_next


class DuelingDeepQNetWork(keras.Sequential):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(DuelingDeepQNetWork, self).__init__()
        activation = 'relu'
        self.dense_x1 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x11 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x12 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x13 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x2 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x3 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x4 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x5 = keras.layers.Dense(fc1_dims, activation=activation)

        self.lstm_x1 = keras.layers.Dense(fc1_dims, activation=activation)
        self.lstm_x2 = keras.layers.Dense(fc1_dims, activation="sigmoid")
        self.lstm_x3 = keras.layers.Dense(fc1_dims, activation="sigmoid")
        self.lstm_x4 = keras.layers.Dense(fc1_dims, activation="tanh")
        self.lstm_x5 = keras.layers.Dense(fc1_dims, activation="sigmoid")

        self.lstm_y1 = keras.layers.Dense(fc2_dims, activation=activation)
        self.lstm_y2 = keras.layers.Dense(fc2_dims, activation="sigmoid")
        self.lstm_y3 = keras.layers.Dense(fc2_dims, activation="sigmoid")
        self.lstm_y4 = keras.layers.Dense(fc2_dims, activation="tanh")
        self.lstm_y5 = keras.layers.Dense(fc2_dims, activation="sigmoid")

        self.dense_y1 = keras.layers.Dense(fc2_dims, activation=activation)
        self.dense_y2 = keras.layers.Dense(fc2_dims, activation=activation)
        self.dense_y3 = keras.layers.Dense(fc2_dims, activation=activation)
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense_x1(state)
        ly = self.lstm_y1(x)
        lx = self.lstm_x1(x)

        x= self.dense_x11(x)
        x= self.dense_x12(x)
        x= self.dense_x13(x)

        lx = self.lstm_x2(lx)
        lx = self.lstm_x3(lx)
        lx = self.lstm_x4(lx)

        ly=self.lstm_y2(ly)
        ly=self.lstm_y3(ly)
        ly=self.lstm_y4(ly)
        ly=self.lstm_y5(ly)

        cc1 = Concatenate(axis=-1)([x, lx, ly])
        x = self.dense_x2(cc1)
        x = self.dense_x3(x)
        x = self.dense_x4(x)
        x = self.dense_x5(x)
        x = self.dense_y1(x)
        x = self.dense_y2(x)
        x = self.dense_y3(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantages(self, state):
        x = self.dense_x1(state)
        lx = self.lstm_x1(x)
        ly = self.lstm_y1(x)
        x = self.dense_x11(x)
        x = self.dense_x12(x)
        x = self.dense_x13(x)

        lx = self.lstm_x2(lx)
        lx = self.lstm_x3(lx)
        lx = self.lstm_x4(lx)


        ly = self.lstm_y2(ly)
        ly = self.lstm_y3(ly)
        ly = self.lstm_y4(ly)
        ly = self.lstm_y5(ly)

        cc1 = Concatenate(axis=-1)([x, lx, ly])
        x = self.dense_x2(cc1)
        x = self.dense_x3(x)
        x = self.dense_x4(x)
        x = self.dense_x5(x)
        x = self.dense_y1(x)
        x = self.dense_y2(x)
        x = self.dense_y3(x)
        A = self.A(x)

        return A


class ReplayBuffer():
    def __init__(self, max_sixe, input_Shape):
        self.mem_size = max_sixe
        self.mem_cntr = 0
        self.input_Shape = input_Shape
        self.state_memory = np.zeros((self.mem_size, *input_Shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_Shape), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.actions_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def save_Transition(self, filename, data_list):
        print("saving transaction {0}".format(filename))
        file_path ="./Memory/"+filename+".txt"
        data_list.tofile(file_path, sep=',', format='%10.5f')
    def load_transition(self, filename):
        print("loading Transaction {0}".format(filename))
        file_path = "./Memory/" + filename + ".txt"
        if path.exists(file_path):
            return np.genfromtxt(file_path, delimiter=',')


    def sample_buffer(self, batch_size):
        max_men = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_men, batch_size, replace=True)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.actions_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, new_states, dones


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, eps_end=0.01,
                 mem_size=1000000, fname="duelingAgent", fc1_dimns=128, fc2_dims=128*2, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_end = eps_end
        self.fname = fname
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, [input_dims])
        print(self.memory.__sizeof__())
        self.q_evale = DuelingDeepQNetWork(fc1_dims=input_dims*2, fc2_dims=input_dims*4, n_actions=n_actions)
        self.q_next = DuelingDeepQNetWork(fc1_dims=input_dims*2, fc2_dims=input_dims*4, n_actions=n_actions)
        self.q_evale.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")


    def observe(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def act(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_evale.advantages(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_evale.get_weights())
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_evale(states)
        q_next = tf.math.reduce_max(self.q_next(new_states), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)
        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx]

        self.q_evale.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        self.learn_step_counter += 1

    def save_model(self):
        print("saving model")
        self.q_evale.save_weights("./save/q_evale/"+self.fname+"_q_evale")
        self.q_next.save_weights("./save/q_next/"+self.fname+"_q_next")



    def load_model(self):
        print("load model")
        self.q_evale.load_weights("./save/q_evale/"+self.fname+"_q_evale")
        self.q_next.load_weights("./save/q_next/"+self.fname+"_q_next")



