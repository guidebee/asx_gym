# OpenAI Australia Stock Exchange (ASX) Gym Env
Open AI Gym Env for Australia Stock Exchange (ASX)
Australia Stock Market Simulations


# Historical stock data
 Download the [SQLite database](https://drive.google.com/open?id=15KkzTrwN38EYPBbKB5wIkDozkKSWSgff) 
 and put in the asx_gym directory
 
 this data contains 10 years historical stock data (updated till 2020-May)

 ```
.
├── LICENSE
├── README.md
├── asx_gym
│   ├── __init__.py
│   ├── db.sqlite3  <--- download db.sqlite3 and put here.
│   └── envs
│       ├── __init__.py

```
# Update stock data and ASX index data 

```bash
  python update_stock_data.py
```

This script retrieves new stock data and ASX index data. the data is updated daily

# Update company Info

some time ,new companies may list on asx ,you may need to run
this scripts to get new companies .

```bash
  python update_company_info.py
```

## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

```bash
  pip install -r requirements.txt
```

Then install this package via

```
pip install -e .
```

## Usage

```python
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


```

![asx gym rendering](https://github.com/guidebee/asx_gym/blob/master/docs/asx_gym_render.png "ASX GYM Rendering")

## Support us at Patreon
[https://www.patreon.com/asx_gym](https://www.patreon.com/asx_gym)

## Tutorials

[ASX Gym Action](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Action)

[ASX Gym Observations](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Observations)

[ASX Gym Reward](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Reward)

[ASX Gym Done Conditions](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Done-Condition)

[ASX Gym Info](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Info)

[ASX Gym Render Modes](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Render-Modes)

[ASX Gym Sample Agents](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Sample-Agents)

[ASX Gym Configurations](https://github.com/guidebee/asx_gym/wiki/ASX-Gym-Configurations)

[View Stock data](https://github.com/guidebee/asx_gym/wiki/View-Stock-data-and-index)

[ASX Simulations Youtube Channels](https://www.youtube.com/channel/UCFefpsZ3xWNhY_JUg6LTu9g)

[ASX Listed Companies](https://github.com/guidebee/asx_gym/wiki/ASX-List-Companies)

