import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta

from datetime import time
from Agent.PG_Dueling_DQN_Agent import Agent as Agent1
import time
pd.options.mode.chained_assignment = None
import Trader as trader
Actions = {
    'hold':0,
    'buy': 1,
    'sell': 2,
    'close': 3,
    'buy_sell': 4,
    'sell_buy': 5,
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

class Live_Env():
    def __init__(self):
        self.symbol = "EURUSD"
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.data =None
        self.balance =0
        self.balance=0
        self.equity=0
        self.account_profit=0
        self.positions_total=0
        self.old_close_position=0
        self.position=Positions["flat"]
        self.new_Close_position=0
        self.last_action="flat"
        self.position_data={}

    def Account_infomation(self):
        acount = self.mt.account_info()._asdict()
        self.balance=acount["balance"]
        self.equity = acount["equity"]
        self.account_profit=acount["profit"]

    def getData(self):
        dataset = self.mt.copy_rates_from_pos(self.symbol, self.mt.TIMEFRAME_M1, 0, 2000)
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})
        dataset.ta.strategy(ta.AllStrategy)
        dataset.drop("time", inplace=True, axis=1)
        dataset.fillna(0.0)
        return dataset

    def scalleDate(self):
        self.data=self.getData()
        if "Name" in self.data:
            self.data.drop('Name', axis=1, inplace=True)
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(self.data)
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized['bid'] = 0
        df_normalized['ask'] = 0
        df_normalized["eqn"] = 0
        df_normalized["balance"] = 0
        df_normalized["account_profit"] = 0
        df_normalized["is_order_placed"] = 0
        observatio=df_normalized.iloc[-1]
        ticket = self.mt.symbol_info_tick(self.symbol)._asdict()
        self.Account_infomation()
        self.new_Close_position = self.data["close"].iloc[-1]
        observatio['bid'] = ticket["bid"]
        observatio['ask'] = ticket["ask"]
        observatio["eqn"] = self.equity
        observatio["balance"] = self.balance
        observatio["account_profit"] = self.account_profit
        self.positions_total = self.mt.positions_total()
        if self.positions_total > 0:
            observatio["is_order_placed"] = 1
            self.positions_total = 1
        else:
            observatio["is_order_placed"] = 0
            self.positions_total = 0
        return observatio

    def Action_Buy(self):
        print("sell")
        if self.positions_total>=1:
            return -1
        else:
            self.position_data=trader.Buy_Action(self.mt, self.symbol)
            self.last_action="long"
            return 0.0005


    def Action_Sell(self):
        print("Buy")
        if self.positions_total>=1:
            return -1
        else:
            self.position_data=trader.Sell_Avtion(self.mt, self.symbol)
            self.last_action="short"
            return 0.5

    def Action_Hold(self):
        print("Hold")
        if self.positions_total>=1:
            return self.account_profit/10
        else:
            return -0.001


    def Action_Close(self):
        print("Closing")
        if self.positions_total>=1:
            trader.Close_Position(self.mt,self.symbol, self.position_data["result"], self.last_action)
            return self.account_profit
        else:
            return -1




    def episode(self, action):

        if self.old_close_position==self.new_Close_position:
            self.old_close_position = self.new_Close_position
            return None
        else:
            if action == 1:
                reward = self.Action_Buy()
            elif action == 2:
                reward = self.Action_Sell()
            elif action == 0:
                reward = self.Action_Hold()
            elif action == 3:
                reward = self.Action_Close()
            elif action == 4:
                reward = self.Action_Close()
                self.positions_total = self.mt.positions_total()
                reward = self.Action_Sell() + reward
            elif action == 5:
                reward = self.Action_Close()
                self.positions_total = self.mt.positions_total()
                reward = self.Action_Buy() + reward
            else:
                reward = -0.1
            print("reward {0}".format(reward))
            self.old_close_position = self.new_Close_position
            return reward





def minute_passed(oldepoch):
    return time.time() - oldepoch >= 60*60


if __name__ == "__main__":
    env = Live_Env()
    observatio = env.scalleDate()
    agent1 = Agent1(n_actions=6, batch_size=64, epsilon=1.00, input_dims=len(observatio), lr=0.003, gamma=0.99)
    agent1.load_model()
    count=0
    start_time = time.time()
    while True:
        action = agent1.act(observatio)
        print(action)
        reward= env.episode(action=action)
        if reward is None:
            print("reward is None")
            observatio = env.scalleDate()

        else:
            new_observatio = env.scalleDate()
            if minute_passed(start_time) and observatio["is_order_placed"]==0:
                agent1.observe(state=observatio, action=action, reward=reward, new_state=new_observatio, done=True)
                agent1.police_learn()
                agent1.save_model()
                start_time=time.time()
            else:
                agent1.observe(state=observatio, action=action, reward=reward, new_state=new_observatio, done=False)
            count+=1
            observatio=new_observatio
            agent1.learn()




