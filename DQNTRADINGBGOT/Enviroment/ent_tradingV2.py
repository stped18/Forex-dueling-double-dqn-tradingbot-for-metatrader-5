import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta
import numpy as np
# from Agent.DDPG.ddpg_torch import Agent
from Agent.TD3.td3_torch import Agent

# from Agent.SAC.sac_torch import Agent

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
        self.lot_size = 1000
        self.riskSize = self.balance * 1.5
        self.last_action = 4
        self.start_price = 0

    def getData(self):
        dataset = self.mt.copy_rates_from_pos(self.symbol, self.mt.TIMEFRAME_M1, 0, self.data_range)
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})
        CustomStrategy = ta.Strategy(
            name="Momo and Volatility",
            description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
            ta=[
                {"kind": "sma", "length": 8},
                {"kind": "sma", "length": 13},
                {"kind": "sma", "length": 21},
                {"kind": "ema", "length": 8},
                {"kind": "ema", "length": 13},
                {"kind": "ema", "length": 21},
                {"kind": "sma", "length": 50},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 50},
                {"kind": "ema", "length": 100},
                {"kind": "ema", "length": 150},
                {"kind": "rvi", },
                {"kind": "bbands", "length": 14, "col_names": ("BBL", "BBM", "BBU")},
                {"kind": "bbands", "length": 21, "col_names": ("BBL", "BBM", "BBU")},
                {"kind": "rsi"},
                {"kind": "tsi"},
                {"kind": "ao"},
                {"kind": "mom"},
                {"kind": "adx"},
                {"kind": "stoch"},
                {"kind": "stochrsi"},
                {"kind": "willr"},
                {"kind": "stdev"},
                {"kind": "short_run"},
                {"kind": "ttm_trend"},
                {"kind": "cci"},
                {"kind": "macd", "fast": 8, "slow": 21},
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
        df_normalized["balance"] = self.balance
        df_normalized["equity"] = self.equity
        df_normalized["is_order_placed"] = 0
        df_normalized["profit"] = self.profit
        df_normalized["ask"] = 0
        df_normalized["bid"] = 0
        df_normalized["step"] = 0
        df_normalized["MAX"] = self.MAX_balance
        df_normalized["MIN"] = self.MIN_balance
        df_normalized["units"] = self.last_action

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
        new_row["step"] = self.steps
        new_row["units"] = self.lot_size
        if self.onTrade:
            orderisplaced = 0
        else:
            orderisplaced = 1
        new_row["balance"] = self.balance
        new_row["equity"] = self.equity
        new_row["is_order_placed"] = orderisplaced
        new_row["profit"] = self.profit
        new_row["MAX"] = self.MAX_balance
        new_row["MIN"] = self.MIN_balance

        # min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        # np_scaled = min_max_scaler.fit_transform(row)
        return new_row

    def Action_Buy(self, row):
        if self.onTrade:
            self.reward += -1 * self.steps
        else:
            self.steps = 0
            self.onTrade = True
            self.action = ACTION["LONG"]
            self.start_price = row["bid"]

    def Action_Sell(self, row):
        if self.onTrade:
            self.reward += -1 * self.steps
        else:
            self.steps = 0
            self.onTrade = True
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
            self.reward += 10
            self.equity = self.balance + self.profit
            self.balance = self.equity
            self.profit = 0
            self.onTrade = False
            self.trade_done = True
            self.steps = 0
            self.action = ACTION["CLOSE"]

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
            self.profit = (observation["ask"] - self.start_price) * self.lot_size
        if self.action == ACTION["SHORT"]:
            self.profit = (self.start_price - observation["bid"]) * self.lot_size
        a = np.argmax(action)
        if a == 0:
            self.Action_Buy(observation)
        if a == 1:
            self.Action_Sell(observation)
        if a == 2:
            self.Action_Close(observation)
        if a == 3:
            self.Action_Hold(observation)
        self.Bonus()
        self.last_time_price = self.time_price
        return (self.reward + self.profit) - (self.steps / 100)

    def Bonus(self):
        if self.balance > self.MAX_balance:
            self.MAX_balance = self.balance * 1.05
            self.MIN_balance = self.balance * 0.95
            self.reward += 100
            self.trade_done = True
        if self.balance < self.MIN_balance:
            self.MAX_balance = self.balance * 1.05
            self.MIN_balance = self.balance * 0.95
            self.reward += -100
            self.trade_done = True
        if self.balance > self.riskSize:
            self.lot_size += 1000
            self.riskSize = self.balance * 1.5


def running(worm_up=1000, data_range=2000, loading=True):
    env = Enviroment(data_range=data_range)
    data = env.getData()
    state = data.iloc[0]
    # agent = Agent(alpha=0.0001, beta=0.001,input_dims=[len(state)], tau=0.001,batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=4)
    agent = Agent(alpha=0.001, beta=0.001, input_dims=[len(state)], tau=0.005, batch_size=100, layer1_size=400,
                  layer2_size=300, env=env, n_actions=4, warmup=(worm_up if loading is False else 1000))
    # agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id="env_id", input_dims=[len(state)], tau=0.005,env=env, batch_size=256, layer1_size=256, layer2_size=256,n_actions=4)

    reward_list = []
    count = 0
    time = 0
    profit = 0
    validation_list = []
    load = False
    a = 0
    if loading:
        agent.load_models()
    while a < 10:
        a += 1
        if loading:
            agent.load_models()
        print("starter loop")
        balance = env.balance
        env.reset()
        if load:
            agent.load_models()
            load = False
        for index, row in data.iterrows():
            time += 1
            action = agent.choose_action(state.to_numpy())
            state = env.Update(row=state)
            reward = env.Step(action=action, observation=state)

            reward_list.append(reward)
            newState = env.Update(row=row)
            agent.remember(state=state.to_numpy(), action=action, reward=reward, new_state=newState.to_numpy(),
                           done=env.trade_done)

            agent.learn()
            state = newState
            if env.trade_done:
                count += 1
                print("Count {4} Balance : {0} reward : {1}  Profit/loss :{2}  steps {3}".format(round(env.balance, 2),
                                                                                                 sum(reward_list),
                                                                                                 round((
                                                                                                                   env.balance - balance),
                                                                                                       2),
                                                                                                 time, count))
                time = 0
                balance = env.balance
                if env.balance > profit:
                    agent.save_models()
                    profit = env.balance
                    load = True
                env.steps = 0
                reward_list = []
                validation_list.append(sum(reward_list))
                time = 0
            if env.balance < 0:
                print("No money")
                reward = -9999999999999999999999999999999999999
                agent.remember(state=state.to_numpy(), action=action, reward=reward, new_state=newState.to_numpy(),
                               done=True)
                return
            env.trade_done = False

        env.reset()


from random import randrange

if __name__ == "__main__":
    count = 0
    loding = False
    while count < 2:
        print("loop nr ", count)
        wormup = randrange(5000, 10000)
        data_range = randrange(wormup, 99999)
        validation_rate = running(worm_up=wormup, data_range=99999, loading=loding)
        loding = True
        count += 1
