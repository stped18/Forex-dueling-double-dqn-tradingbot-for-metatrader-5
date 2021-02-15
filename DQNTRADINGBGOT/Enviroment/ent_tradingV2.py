import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta
import numpy as np
from Agent.DDPG.ddpg_torch import Agent
#from Agent.TD3.td3_torch import Agent

#from Agent.SAC.sac_torch import Agent

ACTION = {"LONG": 1, "SHORT": 0, "CLOSE": 2, "HOLD": 3}
POSITION = {"LONG": 1, "SHORT": 0, "FLAT": 2}


class Enviroment(object):
    def __init__(self, data_range):
        self.symbol = "EURUSD"
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.Testing_Data = None
        self.trade_done = False
        self.steps = 0
        self.units = 100000
        self.data_range = data_range
        self.balance = 100
        self.equity = self.balance
        self.profit = 0
        self.onTrade = False
        self.action = ACTION["CLOSE"]
        self.last_Positon = POSITION["FLAT"]
        self.Positon = POSITION["FLAT"]
        self.time_price = 0
        self.last_time_price = 0
        self.reward = 0
        self.MAX_balance = self.balance * 1.01
        self.MIN_balance = self.balance * 0.75
        self.lot_size = 0.01
        self.riskSize = self.balance * 2
        self.last_action = 4
        self.start_price = 0
        self.trade_done=True

    def getData(self):
        dataset = self.mt.copy_rates_from_pos(self.symbol, self.mt.TIMEFRAME_M1, 0, self.data_range)
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})
        CustomStrategy = ta.Strategy(
            name="Momo and Volatility",
            description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
            ta=[
                {"kind": "sma", "length": 5},
                {"kind": "sma", "length": 8},
                {"kind": "sma", "length": 13},
                {"kind": "ema", "length": 8},
                {"kind": "ema", "length": 13},
                {"kind": "ema", "length": 21},
                {"kind": "ema", "length": 50},
                {"kind": "ema", "length": 150},
                {"kind": "bbands", "length": 14, "col_names": ("BBL", "BBM", "BBU")},
                {"kind": "bbands", "length": 21, "col_names": ("BBL", "BBM", "BBU")},
                {"kind": "rsi", "lenght":21},
                {"kind": "rsi", "lenght":14},
                {"kind": "rsi", "lenght":7},
                {"kind": "mom", "lenght":21},
                {"kind": "mom", "lenght":14},
                {"kind": "mom", "lenght":7},
                {"kind": "cci", "lenght":7},
                {"kind": "cci", "lenght":14},
                {"kind": "cci", "lenght":21},
                {"kind": "stoch","fast_k":5, "slow_d":3,"slow_k":3},
                {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
            ]
        )
        dataset.ta.strategy(CustomStrategy)
        # dataset.drop("time", inplace=True, axis=1)
        dataset.fillna(0.0)

        if "Name" in dataset:
            dataset.drop('Name', axis=1, inplace=True)
        # min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        # np_scaled = min_max_scaler.fit_transform(dataset)
        df_normalized = pd.DataFrame(dataset)
        df_normalized = df_normalized.iloc[500:]
        df_normalized["is_order_placed"] = 0
        df_normalized["profit"] = self.profit
        df_normalized["ask"] = 0
        df_normalized["bid"] = 0
        df_normalized["done"] = 0
        df_normalized["reward"] = 0
        df_normalized["1"] = 0
        df_normalized["2"] = 0
        df_normalized["3"] = 0
        df_normalized["4"] = 0
        df_normalized["5"] = 0
        df_normalized["6"] = 0
        df_normalized["7"] = 0
        df_normalized["8"] = 0
        df_normalized["9"] = 0

        return df_normalized

    def reset(self):
        self.steps = 0
        self.balance = 100
        self.equity = self.balance
        self.profit = 0
        self.OnTrade = ACTION["CLOSE"]
        self.last_Positon = POSITION["FLAT"]
        self.Positon = POSITION["FLAT"]
        self.time_price = 0
        self.last_time_price = 0
        self.reward = 0
        self.close_profit = 0
        self.MAX_balance = self.balance * 1.05
        self.MIN_balance = self.balance * 0.95

    def Update(self, row):
        new_row = row
        new_row["ask"] = row["close"] + (row["spread"] / 1000)
        new_row["bid"] = row["close"]
        if self.onTrade:
            orderisplaced = 0
        else:
            orderisplaced = 1
        new_row["is_order_placed"] = orderisplaced
        new_row["profit"] = self.profit
        new_row["done"] = self.trade_done
        new_row["reward"] = self.reward


        # min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        # np_scaled = min_max_scaler.fit_transform(row)
        return new_row

    def Action_Buy(self, row):
        if self.onTrade:
            self.reward += -1
        else:
            self.steps = 0
            self.onTrade = True
            self.trade_done = False
            self.action = ACTION["LONG"]
            self.start_price = row["bid"]

    def Action_Sell(self, row):
        if self.onTrade:
            self.reward += -1
        else:
            self.steps = 0
            self.onTrade = True
            self.trade_done = False
            self.action = ACTION["SHORT"]
            self.start_price = row["ask"]

    def Action_Hold(self, row):
        if not self.onTrade:
            self.reward += -0.001 * self.steps
        else:
            self.reward += 1
        self.equity = self.balance + self.profit

    def Action_Close(self, row):

        if not self.onTrade:
            self.reward += -1
        else:
            self.reward += 0.01
            self.equity = self.balance + self.profit
            self.balance = self.equity
            self.profit = 0
            self.steps = 0
            self.onTrade=False
            self.action = ACTION["CLOSE"]
            self.trade_done = True

    def Find_Position(self, row):
        self.time_price = (row["ask"] + row["bid"]) / 2
        if self.time_price > self.last_time_price:
            self.Positon = POSITION["LONG"]
        elif self.time_price < self.last_time_price:
            self.Positon = POSITION["SHORT"]
        else:
            self.Positon = POSITION["FLAT"]

    def Step(self, action, observation):
        self.steps += 1
        self.reward = 0
        self.Find_Position(row=observation)
        if self.action == ACTION["LONG"]:
            self.profit = (observation["bid"] - self.start_price) * (self.units*self.lot_size)
        if self.action == ACTION["SHORT"]:
            self.profit = (self.start_price - observation["ask"]) * (self.units * self.lot_size)
        self.equity = self.balance + self.profit
        a = np.argmax(action)
        if a == 0:
            self.Action_Buy(observation)
        if a == 1:
            self.Action_Sell(observation)
        if a == 2:
            self.Action_Hold(observation)
        else:
            self.Action_Close(observation)
        self.Bonus()
        self.last_time_price = self.time_price
        return self.profit+self.reward

    def Bonus(self):
        if self.balance > self.MAX_balance:
            self.MAX_balance = self.balance * 1.05
            self.MIN_balance = self.balance * 0.95
            self.reward += 1
            self.trade_done = True
        if self.balance < self.MIN_balance:
            self.MAX_balance = self.balance * 1.05
            self.MIN_balance = self.balance * 0.95
            self.reward += -1
            self.trade_done = True
        if self.balance > self.riskSize:
            self.lot_size += 0.01
            self.riskSize = self.balance * 2
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
def running(worm_up=1000, data_range=2000, loading=True):
    env = Enviroment(data_range=data_range)
    data = env.getData()
    state = data.iloc[0]
    agent = Agent(alpha=0.0001, beta=0.001,input_dims=[len(state)], tau=0.001,batch_size=64, fc1_dims=800, fc2_dims=300, n_actions=4)
    #agent = Agent(alpha=0.001, beta=0.001, input_dims=[len(state)], tau=0.005, batch_size=100, layer1_size=800,layer2_size=400, env=env, n_actions=4, warmup=1000, symbol="EURUSD")
    #agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id="env_id", input_dims=[len(state)], tau=0.005,env=env, batch_size=256, layer1_size=256, layer2_size=256,n_actions=4)

    reward_list = []
    count = 0
    time = 0
    profit = 0
    validation_list = []
    load = False
    a = 0
    high_reward =0
    #agent.load_models()
    while a < 1:
        a += 1
        b = 0
        env = Enviroment(data_range=99999)
        data = env.getData()
        state = data.iloc[0]
        balance=env.balance
        i=0
        for index, row in data.iterrows():
            time += 1
            b+=1
            action = agent.choose_action(state.to_numpy())
            state = env.Update(row=state)
            reward = env.Step(action=action, observation=state)
            reward_list.append(reward)
            newState = env.Update(row=row)
            agent.remember(state=state.to_numpy(), action=action, reward=reward, new_state=newState.to_numpy(),done=env.trade_done)
            print("nr {0} Balance: {1} Equntit: {2} Prifit {3} bid: {4} ask: {5} action:{6} reward: {7}".format(b, env.balance, env.equity, env.profit, state["bid"], state["ask"], list(ACTION.keys())[list(ACTION.values()).index(env.action)], reward))
            agent.learn()
            state = newState
            i+=1

            if env.trade_done:
                count += 1
                print("Count {4} Balance : {0} reward : {1}  Profit/loss :{2}  steps {3}".format(round(env.balance, 2),
                                                                                                 sum(reward_list),
                                                                                                 round(( env.balance - balance),2),
                                                                                                 time, count))
                balance = env.balance
                if high_reward<reward:


                    high_reward=reward
                if env.balance > profit:
                    profit = env.balance
                    load = True
                env.steps = 0
                agent.save_models()
                reward_list = []
                validation_list.append(sum(reward_list))
                time = 0
            if env.balance < 0:
                print("No money")
                reward = -9999999999999999999999999999999999999
                agent.remember(state=state.to_numpy(), action=action, reward=reward, new_state=newState.to_numpy(),done=True)
                return

            env.trade_done = False




from random import randrange

if __name__ == "__main__":
    count = 0
    loding = False
    while count < 2:
        print("loop nr ", count)
        wormup = randrange(5000, 10000)
        data_range = randrange(wormup, 99999)
        validation_rate = running(worm_up=wormup, data_range=2500, loading=loding)
        loding = True
        count += 1
