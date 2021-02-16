import csv
#from Agent.DDPG.ddpg_torch import Agent
import numpy as np
from Agent.Rainbow.memories import PrioritizedNStepMemory
from Agent.Rainbow.networks import NoisyDuelingNetwork
from Agent.Rainbow.rainbow_agent import DQNAgent



class Enviroment():
    def __init__(self, balance):
        self.balance = balance
        self.done = 0
        self.reward=0
        self.steps=0
        self.profit=0
        self.positionsTotal=0
        self.order ={
            "type":None,
            "price":0,
        }
        self.ask=0
        self.bid=0


    def update(self, state):
        self.bid = state[1]
        self.ask= state[2]
        if self.positionsTotal==1:
            if self.order["type"]=="LONG":
                self.profit = (self.bid-self.order["price"])*1000
            if self.order["type"] == "SHORT":
                self.profit = (self.order["price"]-self.ask) * 1000
        state[0]=self.profit
        state[37]=self.positionsTotal
        state[38]=self.reward
        state[39]=self.done
        state[40]=self.steps
        return state

    def step(self, action, data):
        self.steps+=1
        self.reward=0
        a = action
        if a == 0:
            action_send = "BUY"
            action_send = "CLOSE"
            if self.positionsTotal == 1:
                self.positionsTotal = 0
                self.balance += self.profit
                self.order = {
                    "type": None,
                    "price": 0,
                }
                self.done = 1
                self.reward = self.profit * 10

            if self.positionsTotal==0:
                self.positionsTotal=1
                self.order = {
                    "type": "LONG",
                    "price": self.ask,
                }
                self.steps=0

                self.reward=10+self.profit
            else:
                self.reward=-100*self.steps
        elif a == 1:
            action_send = "SELL"
            action_send = "CLOSE"
            if self.positionsTotal == 1:
                self.positionsTotal = 0
                self.balance += self.profit
                self.order = {
                    "type": None,
                    "price": 0,
                }
                self.done = 1
                self.reward = self.profit * 10

            if self.positionsTotal==0:
                self.positionsTotal=1
                self.order = {
                    "type": "SHORT",
                    "price": self.bid,
                }
                self.steps=0

                self.reward=10+self.profit
            else:
                self.reward=-100*self.steps
        else:
            action_send = "HOLD"
            if self.positionsTotal==1:
                self.reward=self.profit-(self.steps/1000)
            else:
                self.reward=self.profit-self.steps
        return self.reward


def dataReader(path):
    f = open(path)
    csv_f = csv.reader(f)
    datalist = []
    for row in csv_f:
        datalist.append(row)
    f.close()
    return datalist


def Train():
    while True:
        data = dataReader("data.csv")
        state = list(np.float_(data[0]))
        env = Enviroment(1000)
        state = env.update(state)
        mem = PrioritizedNStepMemory(int(1e5))
        agent = DQNAgent(state_size = len(state), hidden_sizes = [64,64],
             action_size = 3, replay_memory = mem,
             double=True, Architecture=NoisyDuelingNetwork)
        reward_list =[]
        higest_reward=0

        for index, item in enumerate(data):
            if item:
                print("running")
                action = agent.act(np.array(state))
                print(action)
                reward = env.step(action=action, data=state)
                reward_list.append(reward)
                new_state = env.update(list(np.float_(item)))
                agent.step(state=np.array(state), action=action, reward=reward, next_state=np.array(new_state), done=False if env.done==0 else True)
                if env.done==1:
                    print("balance {0} Profit {1} sum reward {2}, steps {3} reward {4}".format(env.balance, env.profit, sum(reward_list), env.steps, reward))
                    reward_list=[]
                    f = open("save_log.txt", "r")
                    validation = f.read()
                    f.close()
                    if validation=="Done":
                        f = open("save_log.txt", "w")
                        f.write("Saving")
                        f.close()
                        agent.save_models()
                        f = open("save_log.txt", "w")
                        f.write("Done")
                        f.close()
                    env.done=0
                state=new_state
                env.done=0

Train()