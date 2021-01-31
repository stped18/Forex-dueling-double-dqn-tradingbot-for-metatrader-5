import statistics
from os import path
from tensorflow.keras.models import load_model, save_model
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.models import Sequential



    

class PolicyGradiensNetwork(keras.Model):
    def __init__(self, acount_dims, n_actions, fc1_dims=256, fc2_dims=256 ):
        super(PolicyGradiensNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        activation = 'relu'
        self.acount_Layer1 = Dense(acount_dims, activation=activation)
        self.acount_Layer2 = Dense(self.fc1_dims, activation=activation)
        self.acount_Layer3 = Dense(self.fc1_dims, activation=activation)

        self.predictions_layer1=Dense(self.fc1_dims, activation="hard_sigmoid")
        self.predictions_layer2=Dense(self.fc1_dims, activation="hard_sigmoid")
        self.predictions_layer3=Dense(self.fc1_dims, activation="hard_sigmoid")

        self.fc1 = Dense(self.fc1_dims, activation=activation)
        self.fc2 = Dense(self.fc1_dims, activation=activation)
        self.fc3 = Dense(self.fc1_dims, activation=activation)

        self.fc4 = Dense(self.fc2_dims * 3, activation=activation)
        self.fc5 = Dense(self.fc2_dims* 3, activation=activation)
        self.fc6 = Dense(self.fc2_dims* 3, activation=activation)
        self.outputLayer = Dense(self.n_actions, activation="softmax")

    def call(self, state):
        positionState = state[0]
        acount_state = state[1]
        value = self.fc1(positionState)
        value = self.fc2(value)
        value = self.fc3(value)

        acount_value = self.acount_Layer1(acount_state)
        acount_value = self.acount_Layer2(acount_value)
        acount_value = self.acount_Layer3(acount_value)

        prediction_value = self.predictions_layer1(positionState)
        prediction_value = self.predictions_layer2(prediction_value)
        prediction_value = self.predictions_layer3(prediction_value)

        mergedOut = Add()([acount_value, value, prediction_value])


        value = self.fc4(mergedOut)
        value = self.fc5(value)
        value = self.fc6(value)
        output = self.outputLayer(value)
        return output


class DuelingDeepQNetWork(keras.Sequential):
    def __init__(self, fc1_dims, fc2_dims, n_actions, acount_dims):
        super(DuelingDeepQNetWork, self).__init__()
        activation = 'relu'
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.acount_Layer1 = Dense(acount_dims, activation=activation, name="acount_Layer1")
        self.acount_Layer2 = Dense(self.fc1_dims, activation=activation, name="acount_Layer2")
        self.acount_Layer3 = Dense(self.fc1_dims, activation=activation, name="acount_Layer2")

        self.predictions_layer1 = Dense(self.fc1_dims, activation="hard_sigmoid", name="predictions_layer1")
        self.predictions_layer2 = Dense(self.fc1_dims, activation="hard_sigmoid",name="predictions_layer2")
        self.predictions_layer3 = Dense(self.fc1_dims, activation="hard_sigmoid",name="predictions_layer3")

        self.fc1 = Dense(self.fc1_dims, activation=activation)
        self.fc2 = Dense(self.fc1_dims, activation=activation)
        self.fc3 = Dense(self.fc1_dims, activation=activation)

        self.fc4 = Dense(self.fc2_dims * 3, activation=activation)
        self.fc5 = Dense(self.fc2_dims * 3, activation=activation)
        self.fc6= Dense(self.fc2_dims * 3, activation=activation)
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)

    def call(self, state):
        positionState = state[0]
        acount_state = state[1]
        value = self.fc1(positionState)
        value = self.fc2(value)
        value = self.fc3(value)

        acount_value = self.acount_Layer1(acount_state)
        acount_value = self.acount_Layer2(acount_value)
        acount_value = self.acount_Layer3(acount_value)

        prediction_value = self.predictions_layer1(positionState)
        prediction_value = self.predictions_layer2(prediction_value)
        prediction_value = self.predictions_layer3(prediction_value)

        mergedOut = Add()([acount_value, value, prediction_value])

        value = self.fc4(mergedOut)
        value = self.fc5(value)
        value = self.fc6(value)
        V = self.V(value)
        A = self.A(value)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantages(self, state):
        positionState=state[0]
        acount_state=state[1]
        value = self.fc1(positionState)
        value = self.fc2(value)
        value = self.fc3(value)

        acount_value = self.acount_Layer1(acount_state)
        acount_value = self.acount_Layer2(acount_value)
        acount_value = self.acount_Layer3(acount_value)

        prediction_value = self.predictions_layer1(positionState)
        prediction_value = self.predictions_layer2(prediction_value)
        prediction_value = self.predictions_layer3(prediction_value)

        mergedOut = Add()([acount_value, value, prediction_value])

        value = self.fc4(mergedOut)
        value = self.fc5(value)
        value = self.fc6(value)
        A = self.A(value)

        return A


class ReplayBuffer():
    def __init__(self, max_sixe, input_Shape, acount_input):
        self.mem_size = max_sixe
        self.mem_cntr = 0
        self.input_Shape = input_Shape
        self.state_memory = np.zeros((self.mem_size, *input_Shape), dtype=np.float32)
        self.acount_state_memory=np.zeros((self.mem_size, *acount_input), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_Shape), dtype=np.float32)
        self.acount_new_state_memory = np.zeros((self.mem_size, *acount_input), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state[0]
        self.acount_state_memory[index]= state[1].iloc[0]
        self.new_state_memory[index] = new_state[0]
        self.acount_new_state_memory[index] =  new_state[1].iloc[0]
        self.actions_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def save_Transition(self, filename, data_list):
        print("saving transaction {0}".format(filename))
        file_path = "./Memory/" + filename + ".txt"
        data_list.tofile(file_path, sep=',', format='%10.5f')

    def load_transition(self, filename):
        print("loading Transaction {0}".format(filename))
        file_path = "./Memory/" + filename + ".txt"
        if path.exists(file_path):
            return np.genfromtxt(file_path, delimiter=',')

    def sample_buffer(self, batch_size):
        max_men = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_men, batch_size, replace=True)
        states=[self.state_memory[batch], self.acount_new_state_memory[batch]]
        new_states=[ self.new_state_memory[batch], self.acount_new_state_memory[batch]]
        actions = self.actions_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, new_states, dones


class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, eps_end=0.01,
                 mem_size=10000, fname="duelingAgent", fc1_dimns=128, fc2_dims=128 * 2, replace=100, acount_dims=10):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_choice = batch_size
        self.eps_dec = epsilon_dec
        self.eps_end = eps_end
        self.fname = fname
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step_counter = 0
        self.input_dims = input_dims * 2
        self.input_dims2 = input_dims * 4
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.memory = ReplayBuffer(mem_size, [input_dims], [acount_dims])
        self.q_evale = DuelingDeepQNetWork(acount_dims=acount_dims,fc1_dims=self.input_dims, fc2_dims=self.input_dims2, n_actions=n_actions)
        self.q_next = DuelingDeepQNetWork(acount_dims=acount_dims, fc1_dims=self.input_dims, fc2_dims=self.input_dims2, n_actions=n_actions)
        self.policy = PolicyGradiensNetwork(acount_dims=acount_dims,n_actions=n_actions, fc1_dims=self.input_dims, fc2_dims=self.input_dims2)
        self.policy.compile(optimizer=Adam(lr=lr))
        self.q_evale.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")

    def observe(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        state= [state[0], state[1].iloc[0]]
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def step(self, observation):
        action_list = []
        for i in range(self.replay_choice):
            if i % 2 == 0:
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.action_space)
                    action_list.append(action)
                else:
                    state = np.array([observation[0]])
                    acount_state = np.array([observation[1].iloc[0]])
                    actions = self.q_evale.advantages([state,acount_state])
                    action = tf.math.argmax(actions, axis=1).numpy()[0]
                    action_list.append(action)
            else:
                state = tf.convert_to_tensor([observation[0]], dtype=tf.float32)
                acount_state = tf.convert_to_tensor([observation[1].iloc[0]], dtype=tf.float32)
                probs = self.policy([state,acount_state])
                action_probality = tfp.distributions.Categorical(probs=probs)
                action = action_probality.sample()
                action = action.numpy()[0]
                action_list.append(action)
        action = np.bincount(action_list).argmax()
        return action

    def DQN_learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_evale.get_weights())
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_evale([states[0], states[1]])
        q_next = tf.math.reduce_max(self.q_next([new_states[0],new_states[1]]), axis=1, keepdims=True).numpy()
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
                the_state = tf.convert_to_tensor([state[0]], dtype=tf.float32)
                acount_state = tf.convert_to_tensor([state[1]], dtype=tf.float32)
                probs = self.policy([the_state, acount_state])
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
        self.q_evale.save_weights("E:/Trading_Model/Models/" + self.fname + "_q_evale")
        self.q_next.save_weights("E:/Trading_Model/Models/" + self.fname + "_q_next")
        self.policy.save_weights("E:/Trading_Model/Models/" + self.fname + "_PG")
        #self.save_transaction()

    def save_transaction(self):
        self.memory.save_Transition("E:/Trading_Model/Memory_files/state_memory", self.memory.state_memory)
        self.memory.save_Transition("E:/Trading_Model/Memory_files/new_state_memory", self.memory.state_memory)
        self.memory.save_Transition("E:/Trading_Model/Memory_files/action_memory", self.memory.actions_memory)
        self.memory.save_Transition("E:/Trading_Model/Memory_files/reward_memory", self.memory.reward_memory)
        self.memory.save_Transition("E:/Trading_Model/Memory_files/done_memory", self.memory.terminal_memory)


    def load_model(self):
        print("load model")
        self.q_evale.load_weights("E:/Trading_Model/Models/" + self.fname + "_q_evale")
        self.q_next.load_weights("E:/Trading_Model/Models/" + self.fname + "_q_next")
        self.policy.load_weights("E:/Trading_Model/Models/" + self.fname + "_PG")
        #self.load_Transaction()

    def load_Transaction(self):
        self.memory.state_memory = self.memory.load_transition("E:/Trading_Model/Memory_files/state_memory")
        self.memory.new_state_memory = self.memory.load_transition("E:/Trading_Model/Memory_files/new_state_memory")
        self.memory.actions_memory = self.memory.load_transition("E:/Trading_Model/Memory_files/action_memory")
        self.memory.reward_memory = self.memory.load_transition("E:/Trading_Model/Memory_files/reward_memory")
        self.memory.terminal_memory = self.memory.load_transition("E:/Trading_Model/Memory_files/done_memory")
