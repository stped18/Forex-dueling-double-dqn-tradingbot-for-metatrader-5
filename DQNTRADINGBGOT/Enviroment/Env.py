import pickle

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn import preprocessing
from Model.dueeling_dqn import Agent
import pandas_ta as ta

Actions = {
        'hold': 0,
        'buy': 1,
        'sell': 2,
        'close': 3
    }

Positions = {
    'flat': 0,
    'long': 1,
    'short': 2,
}
lr = 0.01
gamma = 0.99
n_actions = Actions
episodes = 10
batch_size = 32


class enviroment():
    def __init__(self):
        self.symbol="EURUSD"
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.lot = 0.03
        self.data = self.getData(self.symbol)
        self.oldClose = 0
        self.position = Positions["flat"]
        self.old_position=self.position
        self.inputdims = None
        self.reward = 0
        self.last_action = 0
        self.last_order=0
        self.order_action=0
        self.ballances= 100

    def pipscalculator(self):
        digts= self.mt.symbol_info(self.symbol).digits

    def getData(self, Symbol):
        dataset = self.mt.copy_rates_from_pos(Symbol, self.mt.TIMEFRAME_M1, 0, 5000)
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})


        # To run your "Custom Strategy"
        dataset.ta.strategy(ta.AllStrategy)
        dataset.drop("time", inplace=True, axis=1)
        dataset.fillna(0.0)
        return dataset

    def scalleDate(self):
        if "Name" in self.data:
            self.data.drop('Name', axis=1, inplace=True)
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(self.data)
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized['close'] = self.data['close'].values
        return df_normalized

    def episode(self, row, action):
        comment =""
        if row["close"] > self.oldClose:
            self.position = Positions["long"]
        elif row["close"] < self.oldClose:
            self.position = Positions["short"]
        else:
            self.position = Positions["flat"]


        if action == Actions["buy"]:
            if self.last_order==0:
                self.last_order=row["close"]
                comment+="\n action Buy"
                self.last_order = row["close"]
                self.order_action = action
                self.last_action = action
                if self.position == Positions["long"]:
                    if self.old_position== Positions["flat"] or self.old_position== Positions["short"]:
                        self.reward=0.5
                    else:
                        self.reward=-1
            else:
                self.reward=-1

        if action == Actions["sell"]:
            if self.last_order==0:
                self.last_order=row["close"]
                comment += "\n action Sell"
                self.last_order=row["close"]
                self.order_action=action
                self.last_action=action
                if self.position == Positions["short"]:
                    if self.old_position == Positions["flat"] or self.old_position == Positions["long"]:
                        self.reward = 0.5
                    else:
                        self.reward = -1
            else:
                self.reward=-1


        if action == Actions["hold"]:
            comment += "\n action Hold"
            if self.last_order!=0:
                if self.last_action==Actions["buy"]:
                    if self.position==Positions["long"]:
                        self.reward= 0.5
                    if self.position == Positions["short"]:
                        self.reward =-0.2
                if self.last_action==Actions["sell"]:
                    if self.position == Positions["short"]:
                        self.reward = 0.5
                    if self.position==Positions["long"]:
                        self.reward=-0.2
            else:
                self.reward=-0.1


        if action == Actions["close"]:
            if self.last_order!=0:
                comment += "\n action Close"
                if self.last_action==Actions["buy"]:
                    comment += "\n Close Buy order"
                    if self.last_order<row["close"]:
                        comment += "\n Profit"
                        self.reward = (row["close"]-self.last_order)*10
                        self.ballances += (row["close"]-self.last_order)*10
                        self.last_order=0
                    else:
                        comment += "\n Not profit"
                        self.reward= (row["close"]-self.last_order)*10
                        self.ballances += (row["close"]-self.last_order)*10
                        self.last_order = 0
                if self.last_action==Actions["sell"]:
                    comment += "\n Close Sell order"
                    if self.last_order>row["close"]:
                        comment += "\n profit"
                        self.reward= (self.last_order-row["close"])*10
                        self.ballances += (self.last_order-row["close"])*10
                        self.last_order = 0
                    else:
                        comment += "\n Not Profit"
                        self.reward= (self.last_order-row["close"])*10
                        self.ballances += (self.last_order-row["close"])*10
                        self.last_order = 0
            else:
                self.reward=-1


        self.oldClose = row["close"]
        self.old_position=self.position
        return self.reward


if __name__ == '__main__':
    e = enviroment()
    data = e.scalleDate()
    state = data.iloc[0]
    agent = Agent(  n_actions=4,  batch_size=32,epsilon=1.00,input_dims=[len(state)], lr=0.001,gamma=0.95)
    count =0
    reward_sum=0
    action=0
    brainisCreatet=True

    for i in range(1000):
        e.ballances=100
        higest_ballance = 0
        reward_sum = 0
        agent.load_model()
        for index, row in e.scalleDate().iterrows():
            action = agent.act(row)
            reward = e.episode(row, action)
            reward_sum += reward
            agent.observe(state=state, action=action, reward=reward, new_state=row, done=False)
            agent.learn()
            state = row
            if e.ballances>higest_ballance:
                higest_ballance=e.ballances
            if e.ballances<=0:
                break
            #print("_____________________________________"
          #"\nepisode :{0} "
          #"\nreward :{1}"
          #"\naction :{2}"
          #"\nbalance :{3}"
                  #"\nHigest balance :{4}".format(index,
                                  #e.reward,
                                  #(list(Actions.keys())[list(Actions.values()).index(action)]),
                                # e.ballances, higest_ballance))

        print("count: {0} \nbalance: {1} \nreward: {2} \nhigest ballance {3}".format(count, e.ballances, reward_sum , higest_ballance))
        count += 1
        agent.save_model()

