import io
import pathlib
import sqlite3
from datetime import datetime, timedelta, date

import cv2
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
# Data manipulation packages
import pandas as pd
from gym import Env
from gym import spaces, logger
from gym.utils import seeding
from gym.utils.colorize import *

from asx_gym.envs.asx_image_viewer import AsxImageViewer

date_fmt = '%Y-%m-%d'
plt.xticks(rotation=90)


def get_img_from_fig(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AsxGymEnv(Env):
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}

    def __init__(self, **kwargs):
        self.fig, self.ax = plt.subplots()
        self.np_random, seed = seeding.np_random(0)
        seed = seeding.create_seed(32)
        self.seed(seed=seed)
        self.step_day_count = 0

        self.viewer = AsxImageViewer()
        self.min_stock_date = date(2011, 1, 10)
        self.min_stock_seq = 0

        # default values
        self.user_set_start_date = kwargs.get('start_date', self.min_stock_date)
        if self.user_set_start_date < self.min_stock_date:
            self.user_set_start_date = self.min_stock_date
        self.start_date = self.user_set_start_date
        self.display_days = kwargs.get('display_days', 20)
        self.keep_same_company_when_reset = kwargs.get('keep_same_company_when_reset', True)
        self.keep_same_start_date_when_reset = kwargs.get('keep_same_start_date_when_reset', False)
        self.simulate_company_number = kwargs.get('simulate_company_number', -1)
        self.simulate_company_list = kwargs.get('simulate_company_list', None)

        self.initial_fund = kwargs.get('initial_fund', 100000)
        self.expected_fund_increase_ratio = kwargs.get('expected_fund_increase_ratio', 2.0)
        self.expected_fund_decrease_ratio = kwargs.get('expected_fund_decrease_ratio', 0.2)

        # company index start from 1, 0 means empty slot
        self.max_company_number = 3000
        self.max_stock_price = 100000
        self.number_infinite = 10000000
        self.random_start_days = 100
        self.max_transaction_days = 0

        # plot styles
        mc = mpf.make_marketcolors(up='g', down='r',
                                   edge='inherit',
                                   wick={'up': 'blue', 'down': 'orange'},
                                   volume='b',
                                   ohlc='i')
        self.style = mpf.make_mpf_style(base_mpl_style='seaborn-whitegrid', marketcolors=mc)

        # random start date
        offset_days = self.np_random.randint(0, self.random_start_days)
        self.start_date = self.user_set_start_date + timedelta(days=offset_days)

        self.action_space = spaces.Dict(
            {
                "company_id": spaces.Discrete(self.max_company_number),
                "buy_or_sell": spaces.Discrete(3),
                "volume": spaces.Box(np.float32(0), high=np.float32(self.number_infinite), dtype=np.float32),
                "price": spaces.Box(low=np.float32(0), high=np.float32(self.max_stock_price),
                                    shape=(self.max_company_number,), dtype=np.float32),
                "end_batch": spaces.Discrete(2)
            }
        )

        self.observation_space = spaces.Dict(
            {
                "indexes": spaces.Dict({
                    'open': spaces.Box(low=np.float32(0), high=np.float32(self.number_infinite),
                                       dtype=np.float32),
                    'close': spaces.Box(low=np.float32(0), high=np.float32(self.number_infinite),
                                        dtype=np.float32),
                    'high': spaces.Box(low=np.float32(0), high=np.float32(self.number_infinite),
                                       dtype=np.float32),
                    'low': spaces.Box(low=np.float32(0), high=np.float32(self.number_infinite),
                                      dtype=np.float32),

                }
                ),
                "day": spaces.Discrete(self.number_infinite),
                "seconds": spaces.Discrete(24 * 3600),
                "company_count": spaces.Discrete(self.max_company_number),
                "prices:": spaces.Dict({
                    "ask_price": spaces.Box(low=np.float32(0), high=np.float32(self.max_stock_price),
                                            shape=(self.max_company_number,), dtype=np.float32),
                    "bid_price": spaces.Box(low=np.float32(0), high=np.float32(self.max_stock_price),
                                            shape=(self.max_company_number,), dtype=np.float32),
                    "price": spaces.Box(low=np.float32(0), high=np.float32(self.max_stock_price),
                                        shape=(self.max_company_number,), dtype=np.float32)}),

                "portfolio_company_count": spaces.Discrete(self.max_company_number),
                "portfolios": spaces.Dict(
                    {
                        "company_id": spaces.MultiDiscrete([self.max_company_number] * self.max_company_number),

                        "volume": spaces.Box(np.float32(0), high=np.float32(self.number_infinite),
                                             shape=(self.max_company_number,),
                                             dtype=np.float32),
                        "buy_price": spaces.Box(low=np.float32(0), high=np.float32(self.max_stock_price),
                                                shape=(self.max_company_number,), dtype=np.float32),
                        "sell_price": spaces.Box(low=np.float32(0), high=np.float32(self.max_stock_price),
                                                 shape=(self.max_company_number,), dtype=np.float32),
                    }),
                "total_value:": spaces.Box(low=np.float32(0), high=np.float32(self.number_infinite), dtype=np.float32),
                "available_fund:": spaces.Box(low=np.float32(0), high=np.float32(self.number_infinite),
                                              dtype=np.float32)

            }
        )

        print(colorize("Initializing data, it may take a couple minutes,please wait...", 'red'))
        db_file = f'{pathlib.Path().absolute()}/asx_gym/db.sqlite3'
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT min(updated_date) as updated_date from stock_dataupdatehistory")
        updated_date = cur.fetchone()
        updated_date = updated_date[0]
        self.max_stock_date = datetime.strptime(updated_date, date_fmt).date()

        if self.user_set_start_date > self.max_stock_date + timedelta(days=-100):
            self.user_set_start_date = self.min_stock_date + timedelta(days=-100)
            self.start_date = self.user_set_start_date

        self.max_transaction_days = (self.max_stock_date - self.min_stock_date).days
        print(colorize(f"Stock date range from {self.min_stock_date} to {self.max_stock_date}", "blue"))
        print(colorize("reading asx index data", 'blue'))
        self.index_df = pd.read_sql_query(
            'SELECT 0 as Seq,index_date as Date,open_index as Open,close_index as Close,high_index as High,low_index as Low,1 as Volume,'
            '0 as Change '
            'FROM stock_asxindexdailyhistory where index_name="ALL ORD"  order by index_date',
            conn,
            parse_dates={'Date': date_fmt}, index_col=['Date'])

        self.index_df = self.index_df.reset_index()
        self.index_df.Seq = self.index_df.index
        self.index_df = self.index_df.set_index('Date')
        self.index_df.columns = ['Seq', 'Open', 'Close', 'High', 'Low', 'Volume', 'Change']
        init_seq = self.index_df[self.index_df.index == '2011-01-10']
        self.min_stock_seq = init_seq.Seq[0]

        print(f'Asx index records:\n{self.index_df.count()}')
        print(colorize("reading asx company data", 'blue'))
        self.company_df = pd.read_sql_query('SELECT id,name,description,code,sector_id FROM stock_company', conn)
        print(f'Asx company count:\n{self.company_df.count()}')
        print(colorize("reading asx sector data", 'blue'))
        self.sector_df = pd.read_sql_query('SELECT id,name,full_name FROM stock_sector', conn)
        print(f'Asx sector count:\n{self.sector_df.count()}')
        # print(colorize("reading asx stock data, please wait...", 'blue'))
        # self.price_df = pd.read_sql_query(
        #     f'SELECT * FROM stock_stockpricedailyhistory order by price_date', con,
        #     parse_dates={'price_date': date_fmt})
        # print(f'Asx stock data records:\n{self.price_df.count()}')
        conn.close()
        print(colorize("Data initialized", "green"))
        self.seed()

    def history_indexes(self, days=-1):
        pass

    def history_stock_prices(self, days=-1):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._close_fig()
        self.ax.clear()
        self.index_df.loc[
            self.index_df.Seq == self.min_stock_seq + self.step_day_count - 1, "Volume"] = self.np_random.randint(100)
        self.index_df.loc[
            self.index_df.Seq == self.min_stock_seq + self.step_day_count - 1, "Change"] = self.np_random.randint(
            20) - 10
        self.step_day_count += 1
        self._draw_stock()

        done = False
        if self.step_day_count > 50:
            done = True
        return self.step_day_count, 0, done, {}

    def reset(self):
        self._close_fig()
        self.step_day_count = 0
        if not self.keep_same_start_date_when_reset:
            offset_days = self.np_random.randint(0, self.random_start_days)
            self.start_date = self.user_set_start_date + timedelta(days=offset_days)

        self._set_start_date()
        logger.info(f'Reset date to {self.start_date}')

        self._draw_stock()

    def render(self, mode='human'):
        if mode == 'ansi':
            pass
        else:
            img = get_img_from_fig(self.fig)
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                self.viewer.imshow(img)
                return self.viewer.is_open

    def _get_current_display_date(self):
        return self.index_df.iloc[self.min_stock_seq + self.step_day_count
                                  :self.min_stock_seq
                                   + self.step_day_count + 1].index.astype(str)[0]

    def _close_fig(self):
        # try to close exist fig if possible
        try:
            plt.close(self.fig)
        except:
            pass

    def _set_start_date(self):
        start_date_index = self.start_date.strftime(date_fmt)
        # find first available index data point
        init_seq = self.index_df[self.index_df.index >= start_date_index].iloc[:1, ]
        new_start_date = str(init_seq.index[0])
        self.start_date = datetime.strptime(new_start_date, f'{date_fmt} %H:%M:%S').date()
        self.min_stock_seq = init_seq.Seq[0]

    def _draw_stock(self):
        stock_index = self.index_df.iloc[
                      self.min_stock_seq + self.step_day_count
                      - self.display_days:self.min_stock_seq + self.step_day_count]

        display_date = self._get_current_display_date()

        self.fig, self.axes = mpf.plot(stock_index,
                                       type='candle', mav=(2, 4),
                                       returnfig=True,
                                       volume=True,
                                       title=f'OpenAI ASX Gym - ALL ORD Index {display_date}',
                                       ylabel='Index',
                                       ylabel_lower='Total Value',
                                       style=self.style

                                       )
        # logger.info(f'Simulate date:{display_date}')
        ax_c = self.axes[3].twinx()
        changes = stock_index.loc[:, "Change"].to_numpy()
        ax_c.plot(changes, color='g', marker='o', markeredgecolor='red', alpha=0.9)
        ax_c.set_ylabel('Value Change')
