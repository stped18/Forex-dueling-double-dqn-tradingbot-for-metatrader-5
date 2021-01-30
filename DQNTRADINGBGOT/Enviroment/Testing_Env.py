import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta
from googlefinance.get import get_data


Actions = {
    'hold':3,
    'buy': 4,
    'sell': 5,
    'close': 3,
    'buy_sell': 2,
    'sell_buy': 1,
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
        self.lot=lot
        self.units=1000
        self.ordertype=ordertype
        self.alive=True

    def Get_Profit(self, current_price):
        if self.ordertype=="short":
            self.profit_loss=self.start_price-current_price*(self.units*self.lot)
            return self.profit_loss
        else:
            self.profit_loss=current_price-self.start_price
            return self.profit_loss


    def Close_order(self, close_price):
        self.close_price=close_price
        self.alive=False
        if self.ordertype == "short":
            self.profit_loss = self.start_price - close_price*(self.units*self.lot)
        else:
            self.profit_loss = close_price - self.start_price*(self.units*self.lot)

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
        self.balance+=order.profit_loss

    def Get_equant(self, current_price ):
        self.equant=self.balance+self.History[-1].Get_Profit(current_price=current_price)
        return self.equant


class Enviroment():
    def __init__(self, symbol, balance=100):
        self.acount = Acount(balance=balance, lot=0.01)
        self.symbol = symbol
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.Testing_Data=None
        self.old_position_price=0

    def reset(self):
        self.acount = Acount(balance=100, lot=0.01)
        self.symbol = "EURUSD"

    def getData(self):
        dataset = self.mt.copy_rates_from_pos(self.symbol, self.mt.TIMEFRAME_M1, 0, (1440)+500)
        dataset = pd.DataFrame(dataset)
        print(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})
        dataset.ta.strategy(ta.AllStrategy)
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
        #df_normalized['bid'] =0
        #df_normalized['ask'] = 0
        #df_normalized['open'] = 0
        df_normalized['price'] = self.Testing_Data["close"]
        df_normalized["eqn"] = 0
        df_normalized["balance"] = 0
        df_normalized["account_profit"] = 0
        df_normalized["is_order_placed"] = 0
        df_normalized = df_normalized.iloc[500:]
        return df_normalized

    def Update(self ,observation):

        if len(self.acount.History) != 0:
            if self.acount.History[-1].alive:
                observation["is_order_placed"] = 1
                observation["account_profit"] = self.acount.History[-1].Get_Profit(current_price=observation["price"])
            else:
                observation["account_profit"] =0
                observation["is_order_placed"] = 0
            self.acount.Get_equant(observation["price"])

        #observation['bid'] = 0
        #observation['ask'] = 0
        observation["eqn"] = self.acount.equant
        observation["balance"] = self.acount.balance
        return observation

    def Action_Buy(self, buy_price):
        if len(self.acount.History)!=0 and self.acount.History[-1].alive:
            return -1
        else:
            self.acount.Add_order(ordertype="long", startprice=buy_price)
            return 1

    def Action_Sell(self, sell_price):
        if len(self.acount.History)!=0 and self.acount.History[-1].alive :
            return -1
        else:
            self.acount.Add_order(ordertype="short", startprice=sell_price)
            return 1

    def Action_Hold(self, current_price):
        if len(self.acount.History)!=0 and self.acount.History[-1].alive:
            re = self.acount.History[-1].Get_Profit(current_price=current_price)
            return round(re,2)
        else:
            return -1

    def Action_Close(self, current_price):
        if len(self.acount.History)!=0 and self.acount.History[-1].alive :
            self.acount.Close_order(current_price)
            re=self.acount.History[-1].profit_loss
            return round(re,2)

        else:
            return -1


    def episode(self, action, observation):
        if len(self.acount.History)!=0:
            self.acount.Get_equant(observation["price"])
        #if action == 1:
            #reward = self.Action_Buy(observation["price"])
        #elif action == 2:
            #reward = self.Action_Sell(observation["price"])
        if action == 0:
            reward = self.Action_Hold(observation["price"])
        #elif action == 3:
            #reward = self.Action_Close(observation["price"])
        elif action == 1:
            reward = self.Action_Close(observation["price"])
            self.positions_total = self.mt.positions_total()
            reward = self.Action_Sell(observation["price"]) + reward
        elif action == 2:
            reward = self.Action_Close(observation["price"])
            self.positions_total = self.mt.positions_total()
            reward = self.Action_Buy(observation["price"]) + reward
        else:
            reward = -0.1
        self.old_position_price=observation["price"]
        return reward


Actions = {
    'hold':0,
    'buy_sell': 1,
    'sell_buy': 2,
}













