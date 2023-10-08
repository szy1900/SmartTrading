import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import pandas_ta
import torch as th
import talib
import pandas_ta as tapds
import tushare as ts

ts.set_token('b023a9881693e6c92efcfe899b7b95741d8d00b17c3bea53119a3985')
import time
import os
from utilits.dataLoader import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from stable_baselines3 import A2C, PPO
from utilits.testModel import test_model

matplotlib.rcParams['axes.unicode_minus'] = False
# support chinese in matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
style.use('ggplot')


def fuc_longshort_enter(x):
    return 1 if x.values[0] == 100 else -1 if x.values[0] == -100 else 0


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, target, features, rewardSign='Positive'):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.datalength = self.df.shape[0]
        self.df_backup = df.copy()
        self.df_backup['holdAmt'] = 0
        self.totalReward = 0
        self.action_space = spaces.Discrete(3)
        self.target = target
        self.rewardSign = rewardSign
        self.features = features
        self.tradingCost = 0
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float16)
        self.ax = None
        print()

    def step(self, action):
        next_step = self.current_step + 1
        prev_holdAmt = self.holdAmt
        if action == 0:
            self.holdAmt = 0
        elif action == 1:
            self.holdAmt = 1
        else:
            self.holdAmt = -1
        self.df_backup.loc[self.df.index[next_step], 'holdAmt'] = self.holdAmt
        # reward = (self.holdAmt*(self.df.loc[self.df.index[min(next_step,self.df.shape[0]-1)], self.target]-self.df.loc[self.df.index[self.current_step], self.target]))
        reward = self.holdAmt * self.df.loc[self.df.index[self.current_step], self.target]
        # if prev_holdAmt!=self.holdAmt:
        #     # print("pay commission")
        #     reward = reward - self.df.loc[self.df.index[self.current_step],'CLOSE']*1/1000
        #     self.tradingCost += self.df.loc[self.df.index[self.current_step], 'CLOSE'] * 1 / 1000

        # if self.rewardSign == 'Negative':
        #     reward = -reward
        # if reward<0:
        #     reward = reward*1.085
        self.totalReward += reward
        done = next_step == len(self.df.index) - 1
        obs = self.df.iloc[next_step, :][self.features].values
        self.current_step += 1
        return obs, reward, done, {'DT': self.df.index[next_step], 'totalReward': self.totalReward,
                                   'tradingCost': self.tradingCost, 'holdAmt': self.holdAmt}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.net_worth = 0
        self.holdAmt = 0
        self.current_step = 0
        self.totalReward = 0
        return self.df.iloc[self.current_step, :][self.features].values
        # print()

    def render(self, close=False):
        df_plot = self.df_backup.iloc[max(0, self.current_step - 1 - 100):self.current_step + 1, :]
        # halfPos = df_plot[df_plot['holdAmt']==0.5]
        fullPos = df_plot[df_plot['holdAmt'] == 1]
        fullNeg = df_plot[df_plot['holdAmt'] == -1]

        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(1, 1, 1)
            # self.ax = df_plot.plot()
            self.ax.clear()
            self.ax.scatter(df_plot.index, df_plot['CLOSE'], s=100)
            self.ax.scatter(fullPos.index, fullPos['CLOSE'], marker='^', s=80)
            self.ax.scatter(fullNeg.index, fullNeg['CLOSE'], marker='v', s=80)

            plt.title(
                f"current step process is {df_plot.index[-1].year}/{df_plot.index[-1].month}/{df_plot.index[-1].day}")

            plt.show(block=False)
            plt.pause(0.5)
        else:
            self.ax.clear()
            self.ax.plot(df_plot.index, df_plot['CLOSE'])
            # self.ax.scatter(halfPos.index,halfPos['国债10Y'],marker='v',s=50,alpha=0.5)
            self.ax.scatter(fullPos.index, fullPos['CLOSE'], marker='^', s=80)
            self.ax.scatter(fullNeg.index, fullNeg['CLOSE'], marker='v', s=80)

            # xticks rotation
            plt.xticks(rotation=45)
            plt.title(
                f"current step process is {df_plot.index[-1].year}/{df_plot.index[-1].month}/{df_plot.index[-1].day}")
            plt.pause(0.5)
            plt.tight_layout()
            plt.show(block=False)
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'TotalReward: {self.totalReward}')


from typing import Callable


def fuc_enter(x):
    return True if (x[0] <= 0 and x[1] > 0) else False

features = ['T_price:信号','buys_macd','f_RSI','CDLMORNINGDOJISTAR', 'CDLENGULFING', 'CDLDOJISTAR']
# features = ['T_price:信号','buys_macd','f_RSI','CDLMORNINGDOJISTAR','CDLENGULFING','CDLDOJISTAR']

# features = ['T_price:信号',,'f_RSI','CDLMORNINGDOJISTAR', 'CDLENGULFING', 'CDLDOJISTAR']
# selectedList = ['CDLMORNINGDOJISTAR', 'CDLENGULFING', 'CDLDOJISTAR', 'CDLCLOSINGMARUBOZU', 'CDLSHORTLINE','CDLLONGLINE', 'CDLSTICKSANDWICH']
patternList = ['CDLMORNINGDOJISTAR', 'CDLENGULFING', 'CDLDOJISTAR']


def fuc_long_enter(x):
    return True if (x.values[0] == 100) else False


def macd(dfData, target):
    dfData['macd_h'] = tapds.macd(dfData[target])['MACDh_12_26_9']
    dfData['buys_macd'] = dfData['macd_h'].rolling(2).apply(lambda x: fuc_enter(x))
    return dfData


if __name__ == '__main__':
    # df = pd.read_csv('data/AAPL.csv', index_col=0)
    # df = df.sort_values('Date')
    # dataloader = DataLoader()
    # df_XY, df_yield_rate = dataloader.data_loader(10)
    pro = ts.pro_api()
    # 获取富时中国50指数

    sybol = 'SPTSX'
    # sybol = 'FTSE'

    df = pd.read_csv(f'data/idxs/{sybol}.csv', encoding='utf_8_sig', parse_dates=['DT']).set_index('DT')
    # df['DT'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
    # df = df.set_index('DT')
    df.rename(columns={'close': 'CLOSE', 'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW'}, inplace=True)
    df.sort_index(inplace=True, ascending=True)
    # df.to_csv(f'data/{sybol}.csv',encoding='utf_8_sig')
    # time.sleep(2)
    target = 'CLOSE'
    df[f'{target}_MA10'] = talib.MA(df[target].values, timeperiod=10)
    df[f'{target}_MA15'] = talib.MA(df[target].values, timeperiod=15)
    df[f'{target}_MA20'] = talib.MA(df[target].values, timeperiod=20)
    df[f'{target}_MA40'] = talib.MA(df[target].values, timeperiod=40)

    colname = target


    def cal_signal(x):
        if x[f'{colname}_MA10'] >= x[f'{colname}_MA15'] and x[f'{colname}_MA15'] >= x[f'{colname}_MA20']:
            return 1
        elif x[f'{colname}_MA10'] <= x[f'{colname}_MA15'] and x[f'{colname}_MA15'] <= x[f'{colname}_MA20']:
            return -1
        else:
            return 0


    df[f'T_price:信号'] = df.apply(lambda x: cal_signal(x), axis=1)
    # df[f'T_price:信号'] = df[f'T_price:信号'] *-1
    df = macd(df, target)
    df['buys_macd'] = df['buys_macd'].replace(0, np.nan).ffill(limit=2).fillna(0)

    # df[f'{target}_diff4'] = 2 * (df[target].diff(4) > 0).astype(int) - 1
    # df['神奇九转'] = (df[f'{target}_diff4'].rolling(9).sum().abs() >= 9).astype(int) * df[f'{target}_diff4']
    df['f_RSI'] = talib.RSI(df[target], timeperiod=14)
    df['f_RSI'] = df['f_RSI'].map(lambda x: 0 if (x >= 10 and x <= 90) else (-1 if x < 10 else 1))

    cl = df['CLOSE']
    op = df['OPEN']
    hi = df['HIGH']
    lo = df['LOW']
    for candle in patternList:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)
        df[candle] = df[candle].rolling(1).apply(fuc_longshort_enter).replace(0, np.nan).ffill(limit=2).fillna(0)

    df['价格空间_MAN'] = df[target].rolling(20).mean()
    df['价格空间'] = (df[target] - df['价格空间_MAN']).map(lambda x: 1 if x >= 0 else -1)

    df = df[[target] + features].fillna(0)
    df['reward_pct'] = df['CLOSE'].pct_change(1).shift(-1).fillna(0)
    df['Close_diff_shft-1'] = df[target].diff(1).shift(-1).fillna(0)

    df['Close_diff_shft-5'] = df[target].diff(5).shift(-5).fillna(0) / 5
    df_train = df[(df.index <= pd.to_datetime("2019/12/31")) & (df.index >= pd.to_datetime("2016/12/31"))]

    coef = abs(df_train['Close_diff_shft-5'][df_train['Close_diff_shft-5'] > 0].sum() / df_train['Close_diff_shft-5'][
        df_train['Close_diff_shft-5'] < 0].sum())
    if coef < 1:
        k = 1 / coef
        df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: k * x if x > 0 else x)
    else:
        k = coef
        df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: x if x > 0 else k * x)
    # if sybol == 'SPTSX':
    #     df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: 1.26 * x if x > 0 else x)
    # if sybol == 'KS11':
    #     df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: x if x > 0 else 1.05 * x)
    # if sybol == 'IXIC':
    #     df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: x if x > 0 else 1.5 * x)
    df_train = df[(df.index <= pd.to_datetime("2019/12/31")) & (df.index >= pd.to_datetime("2016/12/31"))]
    df_test = df[(df.index > pd.to_datetime("2019/12/31")) & (df.index <= pd.to_datetime("2022/12/31"))]

    env_target = 'Close_diff_shft-5_modified'
    df_train['daily_reward'] = df_train[env_target]
    # df_train['daily_reward'].cumsum().plot(c='b')
    # df_train['Close_diff_shft-1'].cumsum().plot(c='r')
    # plt.show()
    # df_train['Close_diff_shft-5_modified'].cumsum().plot()
    # df_train['Close_diff_shft-5'].cumsum().plot()
    #
    # plt.legend(['Price Trend after Adaptive Scaler', 'Original Price Trend'],fontsize=12)
    # plt.xlabel('Date', fontsize=16)
    # plt.ylabel('Price Change', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(f'ExpFigs/AdaptiveScaler_{sybol}.eps', dpi=800)
    # # plt.savefig(f'ExpFigs/AdaptiveScaler_b.png')
    # plt.show()
    # print()
    plot_sig = 'T_price:信号'
    # plot_sig = patternList[2]
    # plot_sig = 'buys_macd'

    # if plot_sig in features:
    #     df_train['daily_reward'] = df_train[env_target] * df_train[plot_sig]
    #     df_train['daily_reward'].cumsum().plot(c='g')
    #     df_train[env_target].cumsum().plot()
    #
    #     df_test['daily_reward'] = df_test['Close_diff_shft-1'] * df_test[plot_sig]
    #     df_test['daily_reward'].cumsum().plot(c='cyan')
    #     df_test['Close_diff_shft-1'].cumsum().plot(c='y')
    #     plt.show()
    # print()

    env = StockTradingEnv(df_train.copy(), env_target, features)
    # env = DummyVecEnv([lambda: env])
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[20, dict(pi=[20, 20], vf=[20, 20])])

    model = A2C('MlpPolicy', env, verbose=1, n_steps=50, gamma=0.9, policy_kwargs=policy_kwargs,
                tensorboard_log="a2c_tensorboard/")
    # # # print(model.policy)
    start = time.time()
    model.learn(total_timesteps=50000, )
    end = time.time()
    tt  = end - start

    # model.save(f"paperResult\\stocks\\a2c_stock_trading2")
    # model =  A2C.load(f"paperResult\\stocks\\a2c_stock_trading2", print_system_info=True)
    # env_target = 'Close_diff_shft-1'
    # env_eval = StockTradingEnv(df_train.copy(), env_target, features=features)
    # test_model(model, env_eval, trade_obj=sybol)
    env_target = 'reward_pct'
    env_eval = StockTradingEnv(df_test.copy(), env_target, features=features)
    test_model(model, env_eval, trade_obj=sybol, benchmark=True)
    print(f"training time: {tt}")

