import time
import MetaTrader5 as mt
from Enviroment.Testing_Env import Enviroment
from Agent.PG_Dueling_DQN_Agent import Agent
import pandas as pd
import uuid
from sklearn import preprocessing

if __name__ == "__main__":
    number_af_agent_to_train = 5
    list_Env = [None] * number_af_agent_to_train
    list_Agents = [None] * number_af_agent_to_train
    list_reward = [None] * number_af_agent_to_train
    actions_list = [None] * number_af_agent_to_train
    acount_states = [None] * number_af_agent_to_train
    reward_lists = [None] * number_af_agent_to_train
    state_list = [None] * number_af_agent_to_train
    new_state_list = [None] * number_af_agent_to_train
    reward_sum_list = [None] * number_af_agent_to_train
    for i in range(len(list_Env)):
        list_Env[i] = Enviroment(symbol="EURUSD", balance=100000)
        list_Env[i].scalleDate()
        acountstates = {"balance": [list_Env[i].acount.balance],
                        "equent": [list_Env[i].acount.balance],
                        "profit": [list_Env[i].acount.profit],
                        "is_order_placed": [0]}

        data = list_Env[i].scalleDate()
        data = data.iloc[0]
        acountstates = pd.DataFrame.from_dict(acountstates)
        state_list[i] = [data, acountstates]
        list_Agents[i] = Agent(n_actions=6, batch_size=64, epsilon=1.00, input_dims=len(data), lr=0.003, gamma=0.98,
                               acount_dims=len(acountstates.iloc[0]))
        list_Agents[i].id_nr = "AGENT NR {0}".format(i)
        reward_lists[i] = []
    for i in range(len(list_Agents)):
        list_Agents[i].load_model()
    count = 0
    while True:

        print("starter loop")
        for index, row in list_Env[0].scalleDate().iterrows():

            for i in range(len(list_Agents)):
                actions_list[i] = list_Agents[i].step(state_list[i])
                reward_lists[i].append(
                    list_Env[i].episode(action=actions_list[i], observation=state_list[i], index=index))
                #print("agent id {4} reward : {0} balance :{1} action :{2} at {3}".format(reward_lists[i][-1],
                #                                                                         list_Env[i].acount.balance,
                #                                                                         actions_list[i], i,
                #                                                                         list_Agents[i].id_nr))
                acount_states[i] = list_Env[i].Update(index)
                new_state_list[i] = [row, acount_states[i]]

            if count == 1440/4:
                for i in range(len(list_Agents)):
                    list_Agents[i].observe(state=state_list[i], action=actions_list[i], reward=reward_lists[i][-1],
                                           new_state=new_state_list[i], done=True)
                    list_Agents[i].Long_term_Learning()
                    list_Agents[i].Short_term_learning()
                    reward_sum_list[i] = sum(reward_lists[i])
                for i in range(len(reward_lists)):
                    print(i)

                m = max(reward_sum_list)
                agent_index = reward_sum_list.index(m)
                list_Agents[agent_index].save_model()
                print("higest reward agent = {0} balance ={1} reward: {2}  order nr : {3}".format(list_Agents[agent_index].id_nr,
                                                                                  list_Env[agent_index].acount.balance,
                                                                                  m,len(list_Env[agent_index].acount.History) ))
                print()
                for i in range(len(list_Agents)):
                    reward_sum_list[i] = []
                time.sleep(5)
                for i in range(len(list_Agents)):
                    list_Agents[i].load_model()
                count = 0
            else:
                for i in range(len(list_Agents)):
                    list_Agents[i].observe(state=state_list[i], action=actions_list[i], reward=reward_lists[i][-1],
                                           new_state=new_state_list[i], done=False)
                    state_list[i] = new_state_list[i]

            count += 1
        for i in range(len(list_Agents)):
            list_Env[i].reset()
