import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta

from Agent.TD3.td3_torch import Agent


class Order(object):
    def __init__(self, ask_price, bid_Price, lors):
        self.start_ask =ask_price
        self.start_bid = bid_Price
        self.LorS=lors
        self.profit_or_loss=0


    def Calculate_Price_diffrens(self,ask_price, bid_Price):
        if self.LorS=="long":
            self.profit_or_loss = (bid_Price-self.start_ask)*1000
        else:
            self.profit_or_loss = (self.start_bid-ask_price)*1000



class Acount(object):
    def __init__(self , balance):
        self.balance=  balance
        self.profit=0
        self.equity=self.balance
        self.order=None
        self.last_order_pl=0

    def Place_Order(self, ask,bid, LorS):
        self.order=Order(ask,bid,LorS)
    def Close_order(self,ask, bid):
        self.order.Calculate_Price_diffrens(ask_price=ask,bid_Price=bid,)
        self.balance+=self.order.profit_or_loss
        self.profit+=self.order.profit_or_loss
        self.last_order_pl=self.order.profit_or_loss
        self.order=None

    def Update_Account(self,  ask,bid):
        if self.order is not None:
            self.order.Calculate_Price_diffrens(ask_price=ask, bid_Price=bid)
            self.equity = self.balance + self.order.profit_or_loss
        else:
            self.equity=self.balance

class Enviroment(object):
    def __init__(self, data_range):
        self.acount= Acount(balance=10000)
        self.symbol = "EURUSD"
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.Testing_Data = None
        self.trade_done=False
        self.steps=0
        self.data_range=data_range



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
                {"kind": "rvi",},
                {"kind": "bbands", "length": 14, "col_names": ("BBL", "BBM", "BBU")},
                {"kind": "bbands", "length": 21, "col_names": ("BBL", "BBM", "BBU")},
                {"kind": "rsi"},
                {"kind": "tsi"},
                {"kind": "ao"},
                {"kind": "mom"},
                {"kind": "adx"},
                {"kind": "adosc"},
                {"kind": "pvt"},
                {"kind": "stoch"},
                {"kind": "fwma"},
                {"kind": "stochrsi"},
                {"kind": "willr"},
                {"kind": "bop"},
                {"kind": "ohlc4"},
                {"kind": "midpoint"},
                {"kind": "midprice"},
                {"kind": "linreg"},
                {"kind": "median"},
                {"kind": "stdev"},
                {"kind": "short_run"},
                {"kind": "ttm_trend"},
                {"kind": "accbands"},
                {"kind": "pdist"},
                {"kind": "mfi"},
                {"kind": "obv"},
                {"kind": "eom"},
                {"kind": "apo"},
                {"kind": "pvol"},
                {"kind": "vp"},
                {"kind": "cci"},
                {"kind": "macd", "fast": 8, "slow": 21},
                {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
            ]
        )
        dataset.ta.strategy(CustomStrategy)
        #dataset.drop("time", inplace=True, axis=1)
        dataset.fillna(0.0)

        if "Name" in dataset:
            dataset.drop('Name', axis=1, inplace=True)
        #min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        #np_scaled = min_max_scaler.fit_transform(dataset)
        df_normalized = pd.DataFrame(dataset)
        df_normalized = df_normalized.iloc[500:]
        df_normalized["balance"]=self.acount.balance
        df_normalized["equity"]=self.acount.equity
        df_normalized["is_order_placed"]=0
        df_normalized["profit"]=self.acount.profit
        df_normalized["ask"]=0
        df_normalized["bid"]=0
        return df_normalized

    def Update(self, row):
        new_row =row
        new_row["ask"] = row["close"] + (row["spread"] / 1000)
        new_row["bid"] = row["close"]
        self.acount.Update_Account(ask=new_row["ask"], bid=new_row["bid"])
        if self.acount.order is None:
            orderisplaced=0
        else:
            orderisplaced=1
        new_row["balance"] = self.acount.balance
        new_row["equity"] = self.acount.equity
        new_row["is_order_placed"] = orderisplaced
        new_row["profit"] = self.acount.profit
        #min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        #np_scaled = min_max_scaler.fit_transform(row)
        return new_row


    def Action_Buy(self, row):
        if self.acount.order is not None:
            return -1
        else:
            self.trade_done = False
            self.acount.Place_Order(ask=row["ask"], bid=row["bid"], LorS="short" )
            self.acount.last_order_pl=0

            return 0.000001

    def Action_Sell(self, row):
        if self.acount.order is not None:
            return -1
        else:
            self.trade_done = False
            self.acount.Place_Order(ask=row["ask"], bid=row["bid"], LorS="short" )
            self.acount.last_order_pl = 0
            return 0.000001

    def Action_Hold(self, row):
        if self.acount.order is not None:
            self.acount.order.Calculate_Price_diffrens(ask_price=row["ask"], bid_Price=row["bid"])
            re= self.acount.order.profit_or_loss
            self.acount.equity=self.acount.balance+re
            return re/1000
        else:
            return -0.001

    def Action_Close(self, row):
        if self.acount.order is not None:
            self.acount.order.Calculate_Price_diffrens(ask_price=row["ask"], bid_Price=row["bid"])
            re = self.acount.order.profit_or_loss
            self.acount.equity = self.acount.balance + re
            self.acount.Close_order(ask=row["ask"] , bid=row["bid"])
            self.acount.order=None
            self.trade_done=True
            return re
        else:
            return -1

    def Step(self, action, observation):
        self.steps+=1
        if action == 1:
            reward = self.Action_Buy(observation)
        if action == 2:
            reward = self.Action_Sell(observation)
        if action == 3:
            reward = self.Action_Hold(observation)
        if action == 0:
            reward = self.Action_Close(observation)
        if action == 5:
            reward = self.Action_Close(observation)
            reward = self.Action_Sell(observation) + reward
        if action == 4:
            reward = self.Action_Close(observation)
            reward = self.Action_Buy(observation) + reward
        if reward==0:
            reward=-0.01
        return reward

import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
def running(worm_up=1000, data_range=99999):
    env = Enviroment(data_range=data_range)
    data = env.getData()
    state = data.iloc[0]
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[len(state)], tau=0.005, env=env, batch_size=100, layer1_size=600,
                  layer2_size=300, n_actions=6, warmup=worm_up)

    reward_list=[]
    count=0
    time = 0
    validation_list=[]
    #agent.load_models()
    print("starter loop")
    for index, row in data.iterrows():
        count+=1
        time += 1
        action = agent.choose_action(np.float32(state.to_numpy()))
        a = np.argmax(action)
        state=env.Update(row=state)
        reward = env.Step(action=a, observation=state)
        reward_list.append(reward)
        newState = env.Update(row=row)
        agent.remember(state=np.float32(state.to_numpy()), action=action, reward=reward, new_state=np.float32(newState.to_numpy()), done=env.trade_done)
        agent.learn()
        state = newState
        if env.trade_done:
            count+=1
            print("Balance : {0} reward : {1}  Profit/loss :{2}  steps {3}".format(env.acount.balance, sum(reward_list), env.acount.last_order_pl, env.steps))
            env.steps = 0
            reward_list=[]
            validation_list.append(sum(reward_list))
            time=0
        env.trade_done = False
    print("Saving ")
    agent.save_models()
    print("saving done")
    return sum(reward_list)




from random import randrange

if __name__ == "__main__":
    count =0
    while True:
        print("loop nr ",count)
        wormup= randrange(1000, 10000)
        data_range = randrange(wormup, 99999)
        validation_rate=running(worm_up=1000, data_range=2000)
        count+=1