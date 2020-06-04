from datetime import date
from logging import INFO

import gym
import asx_gym
# from agents.buy_and_keep_agent import BuyAndKeepAgent
from agents.random_agent import RandomAgent
from asx_gym.envs import AsxObservation

gym.logger.set_level(INFO)
start_date = date(2019, 5, 1)
simulate_company_list = [2, 3, 4, 5, 6, 44, 300, 67, 100, 200]
# simulate_company_list = [3]
env = gym.make("AsxGym-v0", start_date=start_date,
               simulate_company_list=simulate_company_list)
stock_agent = RandomAgent(env)
# stock_agent = RandomAgent(env, min_volume=100, max_volume=500)
# stock_agent = BuyAndKeepAgent(env, 3)

observation = env.reset()
for _ in range(200000 * 24):
    env.render()
    company_count = len(env.simulate_company_list)

    observation, reward, done, info = env.step(stock_agent.action())
    if done:
        env.insert_summary_images(30)
        observation = env.reset()
        stock_agent.reset()
    if observation is not None:
        asx_observation = AsxObservation(observation)
        print(asx_observation.to_json_obj())
        print(info)

env.close()
