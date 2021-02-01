import gym
from Agent.PG_Dueling_DQN_Agent import Agent
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
    # env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)
    agent = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=10, lr=0.001, gamma=0.95)
    observation = env.reset()
    while True:
        print(observation[0].tolist())
        action = agent.act(observation=observation)
        print(action)
        new_obsavation, reward, done, info = env.step(action)
        agent.observe(state=list(observation), reward=reward, action=action, new_state=list(new_obsavation), done=False)
        observation=new_obsavation
        agent.Short_term_learning()
        # env.render()
        if done:
            print("info:", info)
            agent.observe(state=observation, reward=reward, action=action, new_state=new_obsavation, done=True)
            agent.Long_term_Learning()
            break

    plt.cla()
    env.render_all()
    plt.show()


agent = Agent(n_actions=3, batch_size=64, epsilon=1.00, input_dims=2, lr=0.001,gamma=0.95)
