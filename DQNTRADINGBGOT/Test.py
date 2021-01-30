import MetaTrader5 as mt
from Enviroment.Testing_Env import Enviroment
from Agent.PG_Dueling_DQN_Agent import Agent

if __name__=="__main__":
    env = Enviroment(symbol="EURUSD", balance=100)
    data= env.scalleDate()
    data = data.iloc[500:]
    state = data.iloc[0]
    agent = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.003, gamma=0.98)
    #agent.load_model()
    reward_list=[]
    while True:
        print("Starter")
        count=0
        for index, row in data.iterrows():

            state=env.Update(state)
            action = agent.act(state)
            reward=env.episode(action=action, observation=state)
            reward_list.append(reward)
            agent.observe(state=state, action=action,reward=reward,new_state=row,done=False)
            state=row
            agent.DQN_learn()
            for i in env.acount.History:
                print(i.alive)
                print(i.profit_loss)
            print("ballance :{0} equent :{1} Profit :{2} .reward {3}, action {4}\n Price ".format(env.acount.balance,
                                                                                          env.acount.equant,
                                                                                          env.acount.profit,
                                                                                          reward,
                                                                                   action))

            if count==1440:
                print("day past")
                print(
                    "ballance :{0} equent :{1} Profit :{2} .reward {3}, action {4}\n Price ".format(env.acount.balance,
                                                                                                    env.acount.equant,
                                                                                                    env.acount.profit,
                                                                                                    sum(reward_list),
                                                                                                    action))


                reward_list=[]
                agent.observe(state=state, action=action, reward=reward, new_state=row, done=True)
                agent.save_model()
                agent.PG_learn()
                agent.DQN_learn()
                count=0

            if env.acount.balance<0:
                print("dead!!")
                print(
                    "ballance :{0} equent :{1} Profit :{2} .reward {3}, action {4}\n Price ".format(env.acount.balance,
                                                                                                    env.acount.equant,
                                                                                                    env.acount.profit,
                                                                                                    sum(reward_list),
                                                                                                    action))

                agent.observe(state=state, action=action, reward=reward, new_state=row, done=True)
                agent.DQN_learn()
                reward_list = []
                env.reset()
                break
            count+=1
        agent.save_model()
        env.reset()