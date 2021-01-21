import MetaTrader5 as mt5
import pandas as pd
from sklearn import preprocessing
from Model.dueeling_dqn import Agent
import pandas_ta as ta
from Enviroment.Order import Order

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
        self.equant=0
        self.position = Positions["flat"]
        self.old_position = self.position
        self.inputdims = None
        self.reward = 0
        self.last_action = 0
        self.last_order = None
        self.order_action = 0
        self.ballances = 100



    def getData(self, Symbol):
        dataset = self.mt.copy_rates_from_pos(Symbol, self.mt.TIMEFRAME_M1, 0, 500)
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
        df_normalized['bid'] = self.data['close'].values
        df_normalized['ask'] = df_normalized['bid'].values+(self.data["spread"].values/10000)
        return df_normalized

    def marketData(self, row):
        print("ASK: {0} BID: {1}".format(row["ask"], row["bid"]))

    def episode(self, row, action):

        if row["bid"] > self.oldClose:
            self.position = Positions["long"]
        elif row["bid"] < self.oldClose:
            self.position = Positions["short"]
        else:
            self.position = Positions["flat"]
            return

        if self.last_order is not None:
            self.equant=round(self.ballances+self.last_order.calculate_prifit(new_ask=row["ask"], new_bid=row["bid"]),2)
        else:
            self.equant=self.ballances

        if action == Actions["buy"]:
            if self.last_order is None:
                self.last_order=Order(row["ask"], row["bid"], position="long")
                self.reward=1.0

            else:
                self.reward = -1.0
        if action == Actions["sell"]:
            if self.last_order is None:
                self.last_order = Order(row["ask"], row["bid"], position="bid")
                self.reward=1
            else:
                self.reward = -1.0
        if action == Actions["hold"]:
            if self.last_order is None:
                self.reward =round(self.equant-self.ballances,2)
            else:
                self.reward = -0.1
        if action == Actions["close"]:
            if self.last_order is not None:
                self.equant = round(self.ballances + self.last_order.calculate_prifit(new_ask=row["ask"], new_bid=row["bid"]),2)
                self.reward = round(self.equant - self.ballances, 2)
                self.ballances=self.equant
                self.last_order.close_Position(row["ask"],row["bid"])
                self.last_order=None
            else:
                self.reward = -1.0
        self.oldClose = row["bid"]
        self.old_position = self.position

        return self.reward


if __name__ == '__main__':
    e = enviroment()
    data = e.scalleDate()
    state = data.iloc[0]
    agent = Agent(n_actions=4, batch_size=32, epsilon=1.00, input_dims=[len(state)], lr=0.001, gamma=0.95)
    count = 0
    reward_sum =[]
    action = 0
    brainisCreatet = True

    for i in range(1000):
        e.ballances = 100
        higest_ballance = 0.0
        reward_sum = []
        # agent.load_model()
        for index, row in e.scalleDate().iterrows():
            action = agent.act(row)
            reward = e.episode(row, action)
            agent.observe(state=state, action=action, reward=e.reward, new_state=row, done=False)
            agent.learn()
            state = row
            reward_sum.append(e.reward)
            if e.ballances > higest_ballance:
                higest_ballance = e.ballances
            if e.ballances <= 0:
                break
            #print("_____________________________________"
         #"\nepisode :{0} "
         #"\nreward :{1}"
         #"\naction :{2}"
         #"\nbalance :{3}"
         #"\nHigest balance :{4}"
        #"\n EQu: {5}".format(index,
         #e.reward,
         #(list(Actions.keys())[list(Actions.values()).index(action)]),
         #e.ballances, higest_ballance, e.equant))
        print("count: {0} \nbalance: {1} \nreward: {2} \nhigest ballance {3}".format(count, e.ballances, sum(reward_sum),
                                                                                     higest_ballance))
        count += 1
        agent.save_model()
