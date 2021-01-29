import statistics
from os import path
from tensorflow.keras.models import load_model, save_model
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np



class PolicyGradiensNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(PolicyGradiensNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc1_dims, activation="relu")
        self.fc3 = Dense(self.fc1_dims, activation="relu")
        self.fc4= Dense(self.fc2_dims, activation="relu")
        self.outputLayer = Dense(self.n_actions, activation="softmax")

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.fc3(value)
        value = self.fc4(value)
        output = self.outputLayer(value)
        return output




class DuelingDeepQNetWork(keras.Sequential):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(DuelingDeepQNetWork, self).__init__()
        activation = 'relu'
        self.dense_x1 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x3 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x4 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_x5 = keras.layers.Dense(fc1_dims, activation=activation)
        self.dense_y1 = keras.layers.Dense(fc2_dims, activation=activation)
        self.dense_y2 = keras.layers.Dense(fc2_dims, activation=activation)
        self.dense_y3 = keras.layers.Dense(fc2_dims, activation=activation)
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense_x1(state)
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
        self.replay_choice=batch_size
        self.eps_dec = epsilon_dec
        self.eps_end = eps_end
        self.fname = fname
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step_counter = 0
        self.input_dims= input_dims*2
        self.input_dims2= input_dims*4
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.memory = ReplayBuffer(mem_size, [input_dims])
        self.q_evale = DuelingDeepQNetWork(fc1_dims=self.input_dims, fc2_dims=self.input_dims2, n_actions=n_actions)
        self.q_next = DuelingDeepQNetWork(fc1_dims=self.input_dims, fc2_dims=self.input_dims2, n_actions=n_actions)
        self.policy = PolicyGradiensNetwork(n_actions=n_actions, fc1_dims=self.input_dims, fc2_dims=self.input_dims2)
        self.policy.compile(optimizer=Adam(lr=lr))
        self.q_evale.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")


    def observe(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    def act(self, observation):
        action_list=[]
        for i in range(self.replay_choice):
            if i%2==0:
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.action_space)
                    action_list.append(action)
                else:
                    state = np.array([observation])
                    actions = self.q_evale.advantages(state)
                    action = tf.math.argmax(actions, axis=1).numpy()[0]
                    action_list.append(action)
            else:
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                probs = self.policy(state)
                action_probality = tfp.distributions.Categorical(probs=probs)
                action = action_probality.sample()
                action=action.numpy()[0]
                #(unique, counts) = np.unique(action, return_counts=True)
                #frequencies = np.asarray((unique, counts)).T
                #action=frequencies[0][0]
                action_list.append(action)
        #(unique, counts) = np.unique(action_list, return_counts=True)
        #frequencies = np.asarray((unique, counts)).T
        #print(frequencies)
        action = np.bincount(action_list).argmax()
        return action

    def DQN_learn(self):
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


    def PG_learn(self):
        action = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        reward = tf.convert_to_tensor(self.reward_memory, dtype=tf.float32)
        state = tf.convert_to_tensor(self.state_memory, dtype=tf.float32)

        G = np.zeros_like(reward)
        for i in range(len(reward)):
            G_sum = 0
            discount = 1
            for k in range(i, len(reward)):
                G_sum += reward[k] * discount
                discount *= self.gamma
            G[i] = G_sum
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                actuion_probability = tfp.distributions.Categorical(probs=probs)
                log_prob = actuion_probability.log_prob(action[idx])
                loss += -g * tf.squeeze(log_prob)
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


    def save_model(self):
        print("saving model")
        self.q_evale.save_weights("G:/save/q_evale/"+self.fname+"_q_evale")
        self.q_next.save_weights("G:/save/q_next/"+self.fname+"_q_next")
        self.policy.save("G:/save/PG/"+self.fname+"_PG")

    def save_transaction(self):
        self.memory.save_Transition("state_memory", self.memory.state_memory)
        self.memory.save_Transition("new_state_memory", self.memory.state_memory)
        self.memory.save_Transition("action_memory", self.memory.actions_memory)
        self.memory.save_Transition("reward_memory", self.memory.reward_memory)
        self.memory.save_Transition("done_memory", self.memory.terminal_memory)


    def load_model(self):
        print("load model")
        self.q_evale.load_weights("G:/save/q_evale/"+self.fname+"_q_evale")
        self.q_next.load_weights("G:/save/q_next/"+self.fname+"_q_next")
        self.policy= load_model("G:/save/PG/"+self.fname+"_PG")

    def load_Transaction(self):
        self.memory.state_memory = self.memory.load_transition("state_memory")
        self.memory.new_state_memory = self.memory.load_transition("new_state_memory")
        self.memory.actions_memory = self.memory.load_transition("action_memory")
        self.memory.reward_memory = self.memory.load_transition("reward_memory")
        self.memory.terminal_memory = self.memory.load_transition("done_memory")