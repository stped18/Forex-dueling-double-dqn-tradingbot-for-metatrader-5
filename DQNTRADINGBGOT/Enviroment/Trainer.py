import csv
import codecs
#from Agent.DDPG.ddpg_torch import Agent
from Agent.TD3.td3_torch import Agent
import numpy as np


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
        self.MIN_balance = self.balance *0.99
        self.units=100000
        self.lot=0.01


    def update(self, state):
        self.bid = state[1]
        self.ask= state[2]
        if self.positionsTotal==1:
            if self.order["type"]=="LONG":
                self.profit = round((((state[1]+state[2])/2)-self.order["price"])*(self.units*self.lot),2)
            if self.order["type"] == "SHORT":
                self.profit = round((self.order["price"]-((state[1]+state[2])/2)) * (self.units*self.lot),2)
        state[0]=self.profit
        state[37]=self.positionsTotal
        state[38]=self.reward
        state[39]=self.done
        state[40]=self.steps
        state[61]=self.balance
        return state

    def step(self, action, data):
        self.reward=0
        self.done=0
        a = np.argmax(action)





        self.lot=(1*(self.balance/100))/100
        if a==0:
            action_send = "BUY"



            if self.positionsTotal==0:
                self.positionsTotal=1
                self.order = {
                    "type": "LONG",
                    "price": data[2],
                }



            else:
                self.reward=-100
        elif a==1:
            action_send = "SELL"



            if self.positionsTotal==0:
                self.positionsTotal=1

                self.order = {
                    "type": "SHORT",
                    "price": data[1],
                }



            else:
                self.reward=-1000000
        elif a == 2:
            #action_send = "CLOSE"
            if self.positionsTotal==1:
                self.positionsTotal=0
                self.balance+=self.profit
                self.reward = self.profit * 1000
                if self.balance<self.MIN_balance:
                    self.reward=-9999999999999999
                    self.MIN_balance= self.balance*0.99
                self.order = {
                    "type": None,
                    "price": 0,
                }
                if self.steps >1440:
                    self.done=1
            else:
                self.reward=-100*self.steps
        else:
            action_send = "HOLD"
            self.done=0
            if self.positionsTotal==1:
                self.reward=self.profit
            else:
                self.reward=self.profit

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

    #path = "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Tester/D0E8209F77C8CF37AD8BF550E51FF075/Agent-127.0.0.1-3000/MQL5/Files/EURUSDdata1.csv"
    #csvReader = csv.reader(codecs.open(path,"rU","utf-16"))
    #data=list(csvReader)
    #print(data[0])
    #state = list(np.float_(data[0]))
    env = Enviroment(100)
    #ate = env.update(state)
    #agent = Agent(alpha=0.0001, beta=0.001, input_dims=[62], tau=0.001, batch_size=64, fc1_dims=800,fc2_dims=300, n_actions=3)
    agent = Agent(alpha=0.001, beta=0.001, input_dims=[62], tau=0.005, batch_size=100, layer1_size=800,
                  layer2_size=400, env=env, n_actions=4, warmup=99999, symbol="EURUSD")
    #agent.load_models()
    epi =0
    count =0
    higestBalance=0
    while True:
        print("-"*100)
        print("starter new episode")
        path = "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Tester/D0E8209F77C8CF37AD8BF550E51FF075/Agent-127.0.0.1-3000/MQL5/Files/EURUSDdata"+str(count)+".csv"
        csvReader = csv.reader(codecs.open(path, "rU", "utf-16"))
        data = list(csvReader)
        state = list(np.float_(data[0]))
        state = env.update(state)
        #agent = Agent(alpha=0.0001, beta=0.001, input_dims=[len(state)], tau=0.001, batch_size=64, fc1_dims=800,fc2_dims=300, n_actions=2)
        count+=1
        if count ==35:
            count=0

        reward_list =[]
        higest_reward=0
        print(data[0])
        for index, item in enumerate(data):
            if item:
                env.steps+=1
                state = env.update(state)
                action = agent.choose_action(np.array(state))
                reward = env.step(action=action, data=state)
                if env.balance<=50:
                    reward=-9999999999999999999999999999999999999999999999999999999999


                reward_list.append(reward)
                new_state = env.update(list(np.float_(item)))

                agent.remember(state=np.array(state), action=action, reward=reward, new_state=np.array(new_state), done=False if env.done==0 else True)
                agent.learn()
                if env.done==1:
                    print("balance {0} Profit {1} sum reward {2}, steps {3} reward {4}".format(
                    env.balance, state[0], sum(reward_list), env.steps, reward))
                    env.steps=0
                    env.done=0
                    if env.balance <= 50:
                        agent.save_models()
                        env.balance=100

                        # agent.load_models()
                    if sum(reward_list)>0:
                        higest_reward=sum(reward_list)
                        agent.save_models()
                        print(higest_reward)
                    reward_list = []
                state=new_state
                env.done=0
        agent.load_models()
        if env.balance>higestBalance:
            higestBalance=env.balance

Train()