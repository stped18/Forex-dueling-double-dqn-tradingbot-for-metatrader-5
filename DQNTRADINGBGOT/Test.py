import MetaTrader5 as mt
from Enviroment.Testing_Env import Enviroment
from Agent.PG_Dueling_DQN_Agent import Agent

if __name__=="__main__":
    env = Enviroment(symbol="EURUSD", balance=100)
    data= env.scalleDate()
    state = data.iloc[0]
    agent = Agent(n_actions=6, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.0005, gamma=0.99)
    reward_list=[]
    while True:
        print("Starter")
        for index, row in data.iterrows():

            state=env.Update(state)
            action = agent.act(state)
            reward=env.episode(action=action, observation=state)
            reward_list.append(reward)
            agent.observe(state=state, action=action,reward=reward,new_state=row,done=False)
            state=row
            agent.DQN_learn()
            if index%60==0:
                print("ballance :{0} equent :{1} Profir :{2} .reward {3}, action {4} ".format(env.acount.balance,
                                                                                              env.acount.equant,
                                                                                              env.acount.profit, sum(reward_list),
                                                                                              action))
                reward_list=[]
                agent.observe(state=state, action=action, reward=reward, new_state=row, done=True)
                agent.save_model()
                agent.PG_learn()
            if env.acount.balance<0:
                agent.observe(state=state, action=action, reward=reward, new_state=row, done=True)
                agent.DQN_learn()
                break

        agent.observe(state=state, action=action, reward=reward, new_state=row, done=True)
        agent.DQN_learn()
        agent.save_model()
        env.reset()