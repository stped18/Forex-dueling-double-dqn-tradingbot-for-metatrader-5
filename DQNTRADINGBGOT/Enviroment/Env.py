import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
from Model.dueeling_dqn import Agent
import pandas_ta as ta
from Enviroment.Order import Order
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

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
        self.symbol = "EURUSD"
        self.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        self.mt = mt5
        self.mt.initialize(path=self.path)
        self.lot = 0.03
        self.data = self.getData(self.symbol)
        self.oldClose = 0
        self.equant = 0
        self.position = Positions["flat"]
        self.old_position = self.position
        self.inputdims = None
        self.reward = 0
        self.last_action = 0
        self.last_order = None
        self.order_action = 0
        self.ballances = 100
        self.max_profit = 0
        self.max_loss = 0
        self.num_loss = 0
        self.num_wins = 0
        self.num_orders=0

    def getData(self, Symbol):
        dataset = self.mt.copy_rates_from_pos(Symbol, self.mt.TIMEFRAME_M1, 0, (1440*20))
        dataset = pd.DataFrame(dataset)
        dataset = dataset.rename(columns={'tick_volume': 'volume'})
        dataset.ta.strategy(ta.AllStrategy)
        dataset.drop("time", inplace=True, axis=1)

        dataset.fillna(0.0)
        return dataset

    def scalleDate(self ):

        if "Name" in self.data:
            self.data.drop('Name', axis=1, inplace=True)
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(self.data)
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized['bid'] = self.data['close'].values
        df_normalized['ask'] = df_normalized['bid'].values + (self.data["spread"].values / 100)
        df_normalized["eqn"] = 0
        df_normalized["balance"] = 0
        df_normalized["num_orders"] = 0
        return df_normalized

    def marketData(self, row):
        print("ASK: {0} BID: {1}".format(row["ask"], row["bid"]))

    def update_status(self, row):
        if self.last_order is not None:
            self.equant = self.ballances + self.last_order.calculate_prifit(new_ask=row["ask"], new_bid=row["bid"])
            self.num_orders=1
        else:
            self.num_orders=0


    def episode(self, row, action):

        if row["bid"] > self.oldClose:
            self.position = Positions["long"]
        elif row["bid"] < self.oldClose:
            self.position = Positions["short"]
        else:
            self.position = Positions["flat"]
            return

        if self.last_order is not None:
            self.equant = self.ballances + self.last_order.calculate_prifit(new_ask=row["ask"], new_bid=row["bid"])
        else:
            self.equant = self.ballances

        if action == Actions["buy"]:
            if self.last_order is None:
                self.last_order = Order(row["ask"], row["bid"], position="long")
                self.reward = 1

            else:
                self.reward = -1.0
        if action == Actions["sell"]:
            if self.last_order is None:
                self.last_order = Order(row["ask"], row["bid"], position="short")
                self.reward = 1
            else:
                self.reward = -1.0
        if action == Actions["hold"]:
            if self.last_order is None:
                self.reward = (self.equant - self.ballances)/10
            else:
                self.reward = -0.0001
        if action == Actions["close"]:
            if self.last_order is not None:
                self.equant = self.ballances + self.last_order.calculate_prifit(new_ask=row["ask"], new_bid=row["bid"])
                self.reward = self.equant - self.ballances
                if self.reward > self.max_profit:
                    self.max_profit = self.reward
                if self.reward < self.max_loss:
                    self.max_loss = self.reward
                if self.reward < 0:
                    self.num_loss += 1
                    #self.reward+=-1
                if self.reward > 0:
                    self.num_wins += 1
                    #self.reward+= 1
                self.ballances = self.equant
                self.last_order.close_Position(row["ask"], row["bid"])
                self.last_order = None
            else:
                self.reward = -1.0
        self.oldClose = row["bid"]
        self.old_position = self.position

        return self.reward


if __name__ == '__main__':
    history_data={"count":0, "data": [],"balance":[],"eqn":[],"action":[],"reward":[]}
    e = enviroment()
    data = e.scalleDate()
    state = data.iloc[0]
    print("Creating Agent")
    agent = Agent(n_actions=4, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.01, gamma=0.95)
    count = 0
    action = 0
    brainisCreatet = True
    agent.load_model()
    higest_ballance_oncount=0
    print("Starting Loop")
    for i in range(1000):
        e.ballances = 100
        e.num_loss=0
        e.num_wins=0
        e.max_loss=0
        e.max_profit=0
        higest_ballance = 0.0
        reward_sum = []
        history_data["count"] = count
        history_data["data"]=[]
        history_data["balance"]=[]
        history_data["eqn"]=[]
        history_data["action"]=[]
        history_data["reward"]=[]

        for index, row in data.iterrows():
            e.update_status(row)
            row["eqn"]=e.equant
            row["balance"]=e.ballances
            row["num_orders"]=e.num_orders
            action = agent.act(row)
            reward = e.episode(row, action)
            history_data["data"].append(row["bid"])
            history_data["balance"].append(e.ballances)
            history_data["eqn"].append(row["eqn"])
            history_data["action"].append(action)
            history_data["reward"].append(e.reward)
            if index ==len(data):
                agent.observe(state=state, action=action, reward=e.reward, new_state=row, done=True)
                print("done episode: {0}".format(count))
            else:
                agent.observe(state=state, action=action, reward=e.reward, new_state=row, done=False)
            agent.learn()
            state = row
            reward_sum.append(e.reward)
            if e.ballances > higest_ballance:
                higest_ballance = e.ballances
            if e.ballances <= 0:
                break
        string_data=("\nCount: {0}  "
              "Balance: {1}  "
              "Reward: {2}  "
              "Higest ballance {3}  "
              "Max_lose: {4}  "
              "Max_profit: {5}  "
              "Number of losses: {6}  "
              "Number of wins: {7}".format(count, e.ballances, sum(reward_sum), higest_ballance, e.max_loss,
                                          e.max_profit, e.num_loss, e.num_wins))
        with open("datafile.txt", "a") as text_file:
            text_file.writelines(string_data)

        if higest_ballance_oncount<e.ballances:
            fig, axs = plt.subplots(5)
            axs[0].plot(history_data["data"])
            axs[1].plot(history_data["action"])
            axs[2].plot(history_data["balance"])
            axs[3].plot(history_data["reward"])
            axs[4].plot(history_data["eqn"])
            plt.show()
            higest_ballance_oncount=e.ballances

        count += 1
        print(string_data)
        print("Saving model")
        agent.save_model()
