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


class Agent():
    def __init__(self, lr=0.003, gamma=0.99, n_actions=4, fc1_dims=256, fc2_dims=256):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradiensNetwork(n_actions=self.n_actions, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.policy.compile(optimizer=Adam(lr=self.lr))

    def chose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probality = tfp.distributions.Categorical(probs=probs)
        action = action_probality.sample()
        return action.numpy()[0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
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

    def save(self):
        self.policy.save("./PG/")

    def load(self):
        self.policy = keras.models.load_model("./PG/")
