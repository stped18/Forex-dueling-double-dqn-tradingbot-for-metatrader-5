import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
import pandas_ta as ta
import numpy as np
from datetime import time
import time
pd.options.mode.chained_assignment = None
import Trader as trader


class Live_Env():
    def __init__(self):
        self.symbol = "EURUSD"
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.data =None
        self.reward=0



    def step(self, action):
        a = np.argmax(action)
        if a == 0:
            action_send="BUY"
        elif a == 1:
            action_send = "SELL"
        else:
            action_send = "CLOSE"
        return action_send





