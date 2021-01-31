import time
import MetaTrader5 as mt
from Enviroment.Testing_Env import Enviroment
from Agent.PG_Dueling_DQN_Agent import Agent
import pandas as pd
from sklearn import preprocessing

if __name__=="__main__":
    env1 = Enviroment(symbol="EURUSD", balance=100000)
    env2 = Enviroment(symbol="EURUSD", balance=100000)
    env3 = Enviroment(symbol="EURUSD", balance=100000)
    data= env1.scalleDate()
    data= env2.scalleDate()
    data= env3.scalleDate()
    data = data.iloc[500:]
    data1 = data.iloc[0]
    data2 = data.iloc[0]
    data3 = data.iloc[0]
    acount_state = {"balance": [env1.acount.balance],
                    "equent": [env1.acount.balance],
                    "is_order_placed": [0],
                    "profit": [0]}

    acount_state = pd.DataFrame.from_dict(acount_state)
    state1=[data1, acount_state]
    state2=[data2, acount_state]
    state3=[data3, acount_state]
    #agent = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=len(state1), lr=0.003, gamma=0.98)
    #agent.load_model()
    agent1 = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=len(data1), lr=0.003, gamma=0.98, acount_dims=len(acount_state.iloc[0]))
    agent2 = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=len(data1), lr=0.003, gamma=0.98, acount_dims=len(acount_state.iloc[0]))
    agent3 = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=len(data1), lr=0.003, gamma=0.98, acount_dims=len(acount_state.iloc[0]))
    reward_list1=[]
    reward_list2=[]
    reward_list3=[]
    count=0
    while True:
        #agent1 = agent1.load_model()
        print("starter loop")
        for index, row in data.iterrows():

            action1 = agent1.step(state1)
            action2 = agent2.step(state2)
            action3 = agent3.step(state3)
            reward1=env1.episode(action=action1, observation=state1, index=index)
            reward2=env2.episode(action=action2, observation=state2, index=index)
            reward3=env3.episode(action=action3, observation=state3, index=index)
            reward_list1.append(reward1)
            reward_list2.append(reward2)
            reward_list3.append(reward3)
            acountState1 = env1.Update(index)
            acountState2 = env2.Update(index)
            acountState3 = env3.Update(index)

            new_state1 = [row, acountState1]
            new_state2 = [row, acountState1]
            new_state3 = [row, acountState1]

            if count==120:
                agent1.observe(state=state1, action=action1, reward=reward1, new_state=new_state1, done=True)
                agent2.observe(state=state2, action=action2, reward=reward2, new_state=new_state2, done=True)
                agent3.observe(state=state3, action=action3, reward=reward3, new_state=new_state3, done=True)
                agent1.PG_learn()
                agent2.PG_learn()
                agent3.PG_learn()
                agent1.DQN_learn()
                agent2.DQN_learn()
                agent3.DQN_learn()
                resum1=sum(reward_list1)
                resum2=sum(reward_list2)
                resum3=sum(reward_list3)

                if resum1>resum2 and resum1> resum3:
                    agent1.save_model()
                    print("higest reward agent1 = {0}".format(resum1))
                if resum2>resum1 and resum2> resum3:
                    agent2.save_model()
                    print("higest reward agent2 = {0}".format(resum2))
                if resum3>resum1 and resum3> resum2:
                    agent3.save_model()
                    print("higest reward agent3 = {0}".format(resum3))
                reward_list1 = []
                reward_list2 = []
                reward_list3 = []
                time.sleep(5)
                agent1 = agent1.load_model()
                agent2 = agent2.load_model()
                agent3 = agent3.load_model()
                count=0
            else:
                agent1.observe(state=state1, action=action1, reward=reward1, new_state=new_state1, done=False)
                agent2.observe(state=state2, action=action2, reward=reward2, new_state=new_state2, done=False)
                agent3.observe(state=state3, action=action3, reward=reward3, new_state=new_state3, done=False)
            state1 = new_state1
            state2 = new_state2
            state3 = new_state3
            count+=1
        env1.reset()
        env2.reset()
        env3.reset()