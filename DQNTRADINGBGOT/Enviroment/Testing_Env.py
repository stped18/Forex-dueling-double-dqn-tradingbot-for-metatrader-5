import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta
from googlefinance.get import get_data
import numpy as np


Actions = {
    'hold':np.array([1,0,0,0,0,0]),
    'buy': np.array([0,1,0,0,0,0]),
    'sell': np.array([0,0,1,0,0,0]),
    'close':np.array([0,0,0,1,0,0]),
    'buy_sell': np.array([0,0,0,0,1,0]),
    'sell_buy': np.array([0,0,0,0,0,1]),
}

Positions = {
    'flat': 0,
    'long': 1,
    'short': 2,
}



class Order():
    def __init__(self, start_price, ordertype, lot):
        self.start_price=start_price
        self.close_price=0
        self.profit_loss=0
        self.lot=0.01
        self.units=100000
        self.ordertype=ordertype
        self.alive=True

    def Get_Profit(self, current_price):
        if self.ordertype=="short":
            self.profit_loss=(self.start_price-current_price)*1000
            return self.profit_loss
        else:
            self.profit_loss=(current_price-self.start_price)*1000
            return self.profit_loss


    def Close_order(self, close_price):
        self.close_price=close_price
        self.alive=False
        if self.ordertype == "short":
            self.profit_loss =( self.start_price - close_price)*1000
        else:
            self.profit_loss = (close_price - self.start_price)*1000

class Acount():
    def __init__(self, balance, lot):
        self.balance=balance
        self.equant=self.balance
        self.History=[]
        self.lot =lot
        self.profit=0

    def Add_order(self, startprice, ordertype):
        self.History.append(Order(start_price=startprice, ordertype=ordertype, lot=self.lot))

    def Close_order(self, closePrice):
        order = self.History[-1]
        order.Close_order(closePrice)
        self.balance += order.profit_loss

    def Get_equant(self, current_price ):
        self.equant=self.balance+self.History[-1].Get_Profit(current_price=current_price)
        return self.equant


class Enviroment():
    def __init__(self, symbol, balance):
        self.acount = Acount(balance=balance, lot=0.01)
        self.symbol = symbol
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.time_in_trade=0
        self.mt.initialize(path=self.path)
        self.Testing_Data=None
        self.trade_done=False
        self.old_position_price=0

    def reset(self):
        self.acount = Acount(balance=100000, lot=0.01)
        self.symbol = "EURUSD"

    def getData(self):
        dataset = self.mt.copy_rates_from_pos(self.symbol, self.mt.TIMEFRAME_M1, 0, 99999)
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
                {"kind": "ao"},
                {"kind": "mom"},
                {"kind": "adx"},
                {"kind": "adosc"},
                {"kind": "pvt"},
                {"kind": "stoch"},
                {"kind": "fwma"},
                {"kind": "stochrsi"},
                {"kind": "willr"},
                {"kind": "macd", "fast": 8, "slow": 21},
                {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
            ]
        )
        dataset.ta.strategy(CustomStrategy)
        dataset.drop("time", inplace=True, axis=1)
        dataset.fillna(0.0)
        return dataset


    def scalleDate(self):
        self.Testing_Data=self.getData()
        if "Name" in self.Testing_Data:
            self.Testing_Data.drop('Name', axis=1, inplace=True)
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(self.Testing_Data)
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized = df_normalized.iloc[500:]
        df_normalized["balance"]=0
        df_normalized["equent"]=0
        df_normalized["is_order_placed"]=0
        df_normalized["profit"]=0
        df_normalized["ask"]=0
        df_normalized["bid"]=0
        df_normalized["price"]=0
        self.Testing_Data=df_normalized
        return df_normalized

    def addAskBid(self,row, index):
        row["ask"] = self.Testing_Data["close"].iloc[index]+(self.Testing_Data["spread"].iloc[index]/1000)
        row["bid"] = self.Testing_Data["close"]
        return row
    def Update(self, index):
        acount_state={"balance": [self.acount.balance],
                     "equent": [self.acount.balance],
                     "is_order_placed":[0.0],
                     "profit":[0.0]}
        if len(self.acount.History) != 0:
            if self.acount.History[-1].alive:
                acount_state["is_order_placed"]=[1]
                acount_state["profit"] = [self.acount.History[-1].Get_Profit(current_price=self.Testing_Data["close"].iloc[index])]
                acount_state["equent"]=[self.acount.Get_equant(self.Testing_Data["close"].iloc[index])]
            else:
                acount_state["is_order_placed"]=[0]

        data = pd.DataFrame.from_dict(acount_state)
        return data


    def Action_Buy(self, buy_price):
        if not len(self.acount.History) == 0 and self.acount.History[-1].alive:
            self.trade_done = False
            return -1
        else:
            self.acount.Add_order(ordertype="long", startprice=buy_price)
            return 0.000001

    def Action_Sell(self, sell_price):
        if not len(self.acount.History) == 0 and self.acount.History[-1].alive :
            self.trade_done = False
            return -1
        else:
            self.acount.Add_order(ordertype="short", startprice=sell_price)
            return 0.000001

    def Action_Hold(self, current_price):
        if not len(self.acount.History) == 0 and self.acount.History[-1].alive:
            re = self.acount.History[-1].Get_Profit(current_price=current_price)
            return round(re,2)/100
        else:
            return -0.1

    def Action_Close(self, current_price):
        if not len(self.acount.History) == 0 and self.acount.History[-1].alive :
            self.acount.Close_order(current_price)
            re=self.acount.History[-1].profit_loss
            self.trade_done = True
            if re==0.0:
                return -0.001
            else:
                if re>1:
                    return round(re,2)*2
                else:
                    return round(re, 2)

        else:
            return -1


    def episode(self, action, observation, index):
        reward=-0.01
        if len(self.acount.History)!=0:
            self.acount.Get_equant(self.Testing_Data["close"].iloc[index])
        if action == 1:
            reward = self.Action_Buy(self.Testing_Data["bid"].iloc[index])
        if action == 2:
            reward = self.Action_Sell(self.Testing_Data["ask"].iloc[index])
        if action == 3:
            reward = self.Action_Hold(self.Testing_Data["close"].iloc[index])
        if action == 0:
            reward = self.Action_Close(self.Testing_Data["close"].iloc[index])
        if action == 5:
            reward = self.Action_Close(self.Testing_Data["close"].iloc[index])
            self.positions_total = self.mt.positions_total()
            reward = self.Action_Sell(self.Testing_Data["close"].iloc[index]) + reward
        if action == 4:
            reward = self.Action_Close(self.Testing_Data["close"].iloc[index])
            self.positions_total = self.mt.positions_total()
            reward = self.Action_Buy(self.Testing_Data["close"].iloc[index]) + reward


        self.old_position_price=self.Testing_Data["close"].iloc[index]
        return reward


Actions = {
    'hold':0,
    'buy_sell': 1,
    'sell_buy': 2,
}













