# asx_gym
Open AI Gym Env for Australia Stock Exchange (ASX)


# Historical stock data
 Download the [SQLite database](https://github.com/asxgym/asx_data/raw/master/db.sqlite3) 
 and put in the asx_gym directory
 
 this data contains 10 years historical stock data (updated till 2020-May-07)

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

