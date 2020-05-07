# asx_gym
Open AI Gym Env for Australia Stock Exchange (ASX)

# data directory
## company
  - sectors.json list all sectors
  - companies.json list all companies
  
## index
  store all ASX index based on date
  
  [code, index_date, index_open, index_close, index_high, index_low]
  

  
## price
  store all price based on date
  
  [code, price_date, price_open, price_close, price_high, price_low, stock_volume]
  
## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import asx_gym

env = gym.make('AsxGym-v0')
```

