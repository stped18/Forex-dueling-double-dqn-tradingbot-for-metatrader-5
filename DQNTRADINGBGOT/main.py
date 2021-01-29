import matplotlib.pylab as plt
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from Agent.PG_Dueling_DQN_Agent import Agent


def Train():
    style.use('fivethirtyeight')
    History = {"bid": [],
               "ask": [],
               "action": [],
               "balance": []}
    e = enviroment()
    data = e.scalleDate()
    state = data.iloc[0]
    data = data.iloc[500:]
    print("Creating Agent")
    agent = Agent(n_actions=4, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.01, gamma=0.95)
    agent1 = Agent1(n_actions=6, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.003, gamma=0.99)
    agent1.load_model()
    count = 0
    action = 0
    brainisCreatet = True

    print("Starting Loop")
    for i in range(1000):
        e.ballances = 100
        e.num_loss = 0
        e.num_wins = 0
        e.max_loss = 0
        e.max_profit = 0
        e.reset()
        e.Testing_Data = e.getData("EURUSD")
        data = e.scalleDate()
        state = data.iloc[0]
        data = data.iloc[500:]
        higest_ballance = 0.0
        reward_sum = []
        agent.load_model()
        History["bid"] = []
        History["ask"] = []
        History["balance"] = []
        History["action"] = []
        back_learn = False

        for index, row in data.iterrows():
            e.update_status(row)
            row["eqn"] = e.equant
            row["balance"] = e.ballances
            row["num_orders"] = e.num_orders
            action = agent.act(row)
            reward = e.episode(row, action)
            if index == len(data):
                agent.observe(state=state, action=action, reward=e.reward, new_state=row, done=True)
                print("done episode: {0}".format(count))
            row["balance"] = e.balance
            row["is_order_placed"] = e.order_is_placed
            row["profit"] = e.profit
            action1 = agent1.act(state)
            if e.order_is_placed == 1 and action == 3:
                back_learn = True
            else:
                agent.observe(state=state, action=action, reward=e.reward, new_state=row, done=False)
                back_learn = False

            reward1 = e.episode(state, action1)
            agent1.observe(state=state, reward=reward1, action=action1, new_state=row, done=False)
            History["bid"].append(state["bid"])
            History["ask"].append(state["ask"])
            History["balance"].append(e.balance)
            History["action"].append(action1)
            agent1.learn()
            if back_learn:
                agent1.observe(state=state, reward=reward1, action=action1, new_state=row, done=True)

            agent.learn()
            state = row
            reward_sum.append(e.reward)
            if e.ballances > higest_ballance:
                higest_ballance = e.ballances
            if e.ballances <= 0:
                reward_sum.append(reward1)
            if e.balance > higest_ballance:
                higest_ballance = e.balance
            if e.balance <= 0:
                break
        string_data = ("\nCount: {0}  "
                       "Balance: {1}  "


def Test():
    style.use('fivethirtyeight')
    e = enviroment()
    data = e.scalleDate()
    state = data.iloc[0]
    data = data.iloc[500:]
    History=[]
    agent = Agent(n_actions=4, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.01, gamma=0.99)
    agent.load_model()
    e.reset()
    for index, row in data.iterrows():
        e.update_status(row)
        row["eqn"] = e.equant
        row["balance"] = e.balance
        row["is_order_placed"] = e.order_is_placed
        row["profit"] = e.profit
        action = agent.act(row)
        stat_action = action
        if e.order_is_placed == 1:
            if action == 1 or action == 2:
                stat_action = 0
        else:
            if action == 3:
                stat_action = 0
        reward = e.episode(row, action)
        agent.observe(state=state, action=action, reward=reward, new_state=row, done=False)
        agent.learn()
        state = row
        History["bid"].append(row["bid"])
        History["ask"].append(row["ask"])
        History["balance"].append(e.balance)
        History["action"].append(stat_action)

    agent.save_model()
    df = pd.DataFrame.from_dict(History)
    df['Buy_ind'] = np.where((df["action"] == 1), 1, 0)
    df['Sell_ind'] = np.where((df["action"] == 2), 1, 0)
    df['Close'] = np.where((df["action"] == 3), 1, 0)
    fig, axs = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5, forward=True)

    axs[0].plot(df["ask"], label="ask", linewidth=0.5)
    axs[0].plot(df["bid"], label="bid", linewidth=0.5)

    axs[0].scatter(df.loc[df['Buy_ind'] == 1, 'bid'].index, df.loc[df['Buy_ind'] == 1, 'bid'].values,
                   label='Buy', color='green', s=25, marker="^")
    axs[0].scatter(df.loc[df['Sell_ind'] == 1, 'bid'].index, df.loc[df['Sell_ind'] == 1, 'bid'].values,
                   label='Sell', color='red', s=25, marker="v")
    axs[0].scatter(df.loc[df['Close'] == 1, 'bid'].index, df.loc[df['Close'] == 1, 'bid'].values,
                   label='close', color='blue', s=25, marker="_")

    # axs[1].plot(History["action"], label="action")
    axs[1].plot(df["balance"], label="balance", linewidth=0.5)
    plt.legend(loc="upper left")
    plt.show()

    print("finish")


if __name__ == "__main__":

    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Please install GPU version of TF")
    while True:
        print("starter traning")
        Train()

