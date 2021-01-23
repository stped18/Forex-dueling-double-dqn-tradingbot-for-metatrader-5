from Enviroment.Env import enviroment
from Model.dueeling_dqn import Agent
import time
import _thread


def Train():
    e = enviroment()
    data = e.scalleDate()
    state = data.iloc[0]
    print("Creating Agent")
    agent = Agent(n_actions=4, batch_size=64, epsilon=1.00, input_dims=len(state), lr=0.01, gamma=0.95)
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
        higest_ballance = 0.0
        reward_sum = []
        agent.load_model()
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
            else:
                agent.observe(state=state, action=action, reward=e.reward, new_state=row, done=False)

            agent.learn()
            state = row
            reward_sum.append(e.reward)
            if e.ballances > higest_ballance:
                higest_ballance = e.ballances
            if e.ballances <= 0:
                break
        string_data = ("\nCount: {0}  "
                       "Balance: {1}  "
                       "Reward: {2}  "
                       "Higest ballance {3}  "
                       "Max_lose: {4}  "
                       "Max_profit: {5}  "
                       "Number of losses: {6}  "
                       "Number of wins: {7}".format(count, e.ballances, sum(reward_sum), higest_ballance,
                                                    e.max_loss,
                                                    e.max_profit, e.num_loss, e.num_wins))
        with open("datafile.txt", "a") as text_file:
            text_file.writelines(string_data)
        count += 1
        print(string_data)
        print("Saving model")
        agent.save_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Train()