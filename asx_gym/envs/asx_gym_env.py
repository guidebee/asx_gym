import numpy as np
import io
from time import sleep
from datetime import datetime, timedelta
import cv2
from gym.envs.classic_control import rendering
from matplotlib.image import imread
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding
import sqlite3
from gym.utils.colorize import *
# Data manipulation packages
import pandas as pd
import pathlib
import mplfinance as mpf

date_fmt = '%Y-%m-%d'

plt.xticks(rotation=90)


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, papertype='a4')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AsxGymEnv(gym.Env):

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.step_count = 0

        self.viewer = rendering.SimpleImageViewer()
        print(colorize("Initializing data, it may take a couple minutes,please wait...", 'red'))
        db_file = f'{pathlib.Path().absolute()}/asx_gym/db.sqlite3'
        con = sqlite3.connect(db_file)
        print(colorize("reading asx index data", 'blue'))
        self.index_df = pd.read_sql_query(
            'SELECT index_date as Date,open_index as Open,close_index as Close,high_index as High,low_index as Low FROM stock_asxindexdailyhistory where index_name="ASX100"  order by index_date',
            con,
            parse_dates={'Date': date_fmt}, index_col=['Date'])

        print(f'Asx index records:\n{self.index_df.count()}')
        print(colorize("reading asx company data", 'blue'))
        self.company_df = pd.read_sql_query('SELECT id,name,description,code,sector_id FROM stock_company', con)
        print(f'Asx company count:\n{self.company_df.count()}')
        print(colorize("reading asx sector data", 'blue'))
        self.sector_df = pd.read_sql_query('SELECT id,name,full_name FROM stock_sector', con)
        print(f'Asx sector count:\n{self.sector_df.count()}')
        # print(colorize("reading asx stock data, please wait...", 'blue'))
        # self.price_df = pd.read_sql_query(
        #     f'SELECT * FROM stock_stockpricedailyhistory order by price_date', con,
        #     parse_dates={'price_date': date_fmt})
        # print(f'Asx stock data records:\n{self.price_df.count()}')
        con.close()
        print(colorize("Data initialized", "green"))
        self.start_date = datetime(2012, 1, 1)
        self.display_days = 60

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.ax.clear()

        self.step_count += 1
        # start_date = self.start_date + timedelta(days=self.step_count)
        # end_date = self.start_date + timedelta(days=self.step_count + self.display_days)
        #
        # self.ax.set_xlim(start_date, end_date)
        # self.ax.set_ylim(0, 10000)
        # self.ax.set_xlabel("Date")
        #
        # self.ax.set_ylabel("Index")
        # self.ax.clear()
        #
        # stock_index = self.index_df.loc[start_date:end_date]
        #
        # self.ax.plot(stock_index.index, stock_index['Close'])
        # self.ax.set_title(f"Asx Index {self.step_count}")

        start_date = self.start_date + timedelta(days=self.step_count)
        end_date = self.start_date + timedelta(days=self.step_count + self.display_days)
        stock_index = self.index_df.loc[start_date:end_date]
        self.fig, self.ax = mpf.plot(stock_index, type='candle',returnfig=True)
        # self.ax.set_xlim(start_date, end_date)
        # self.ax.set_ylim(0, 10000)
        # self.ax.set_xlabel("Date")
        # self.ax.set_ylabel("Index")
        #
        # self.ax.set_title(f"Asx Index {self.step_count}")

        done = False
        if self.step_count > 360:
            done = True
        return self.step_count, 0, done, {}

    def reset(self):

        self.step_count = 0
        start_date = self.start_date + timedelta(days=self.step_count)
        end_date = self.start_date + timedelta(days=self.step_count + self.display_days)
        stock_index = self.index_df.loc[start_date:end_date]
        self.fig, self.ax = mpf.plot(stock_index, type='candle',returnfig=True)
        # self.ax.set_xlim(start_date, end_date)
        # self.ax.set_ylim(0, 10000)
        # self.ax.set_xlabel("Date")
        # self.ax.set_ylabel("Index")
        #
        # self.ax.set_title(f"Asx Index {self.step_count}")

    def render(self, mode='human'):
        img = get_img_from_fig(self.fig)
        plt.close(self.fig)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            self.viewer.imshow(img)
            return self.viewer.isopen
