import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta


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



class Order():
    def __init__(self, start_price, ordertype):
        self.start_price=start_price
        self.close_price=0
        self.profit_loss=0
        self.ordertype=ordertype
        self.alive=True

    def Get_Profit(self, current_price):
        if self.ordertype=="short":
            self.profit_loss=self.start_price-current_price
            return self.profit_loss
        else:
            self.profit_loss=current_price-self.start_price
            return self.profit_loss


    def Close_order(self, close_price):
        self.close_price=close_price
        self.alive=False
        if self.ordertype == "short":
            self.profit_loss = self.start_price - close_price
        else:
            self.profit_loss = close_price - self.start_price

class Acount():
    def __init__(self, balance, lot):
        self.balance=balance
        self.equant=self.balance
        self.History=[]
        self.lot =lot
        self.profit=0

    def Add_order(self, startprice, ordertype):
        self.History.append(Order(start_price=startprice, ordertype=ordertype))

    def Close_order(self, closePrice):
        order = self.History[-1]
        order.close_order(closePrice)
        self.balance+order.profit_loss

    def Get_equant(self, current_price ):
        self.equant=self.balance+self.History[-1].Get_Profit(current_price=current_price)
        return self.equant


class Enviroment():
    def __init__(self, symbol):
        self.acount = Acount(balance=1000, lot=0.01)
        self.symbol = symbol
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.Testing_Data=self.getData()

    def getData(self):
        dataset = self.mt.copy_rates_from_pos(self.symbol, self.mt.TIMEFRAME_M1, 0, 2000)
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})
        dataset.ta.strategy(ta.AllStrategy)
        dataset.drop("time", inplace=True, axis=1)
        dataset.fillna(0.0)
        return dataset


    def scalleDate(self):
        if "Name" in self.Testing_Data:
            self.Testing_Data.drop('Name', axis=1, inplace=True)
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(self.Testing_Data)
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized['bid'] =0
        df_normalized['ask'] = 0
        df_normalized['open'] = 0
        df_normalized['close'] = 0
        df_normalized["eqn"] = 0
        df_normalized["balance"] = 0
        df_normalized["account_profit"] = 0
        df_normalized["is_order_placed"] = 0
        return df_normalized

    def Update(self ,observation):
        if self.acount.History[-1].alive:
            observation["is_order_placed"] = 1
            observation["account_profit"] = self.acount.History[-1].Get_Profit(current_price=self.data)
        else:
            observation["account_profit"] =0
            observation["is_order_placed"] = 0
        observation['bid'] = 0
        observation['ask'] = 0
        observation["eqn"] =self.acount.equant
        observation["balance"] = self.acount.balance

        observation["is_order_placed"] = 0

    def Action_Buy(self):


    def Action_Sell(self):


    def Action_Hold(self):


    def Action_Close(self):


    def episode(self, action, observation):
        if self.acount.History[-1].alive:
            observation["is_order_placed"] = 1
        else:
            observation["is_order_placed"] = 0




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
        return reward
















