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

import os
from utilits.dataLoader import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from stable_baselines3 import A2C,PPO
from utilits.testModel import test_model
matplotlib.rcParams['axes.unicode_minus'] = False
# support chinese in matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
style.use('ggplot')




def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span=fast, adjust=False).mean()
    exp2 = price.ewm(span=slow, adjust=False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns={'CLOSE': 'macd'})
    signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={'macd': 'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns={0: 'hist'})
    frames = [pd.DataFrame(price), macd, signal, hist]
    df = pd.concat(frames, axis=1)
    return df

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, df,target,features, rewardSign='Positive'):
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
        if action ==0:
            self.holdAmt = 0
        elif action ==1:
            self.holdAmt = 1
        else:
            self.holdAmt = -1
        self.df_backup.loc[self.df.index[next_step], 'holdAmt'] = self.holdAmt
        # reward = (self.holdAmt*(self.df.loc[self.df.index[min(next_step,self.df.shape[0]-1)], self.target]-self.df.loc[self.df.index[self.current_step], self.target]))
        reward = self.holdAmt*self.df.loc[self.df.index[self.current_step], self.target]
        # sigs = self.df.iloc[self.current_step, :][self.features]
        # if 'buys_macd' in sigs:
        #     if sigs['buys_macd'] == 1:
        #         reward = reward*5
        # if prev_holdAmt!=self.holdAmt:
        #     # print("pay commission")
        #     reward = reward - self.df.loc[self.df.index[self.current_step],'CLOSE']*1/1000
        #     self.tradingCost += self.df.loc[self.df.index[self.current_step], 'CLOSE'] * 1 / 1000

        # if self.rewardSign == 'Negative':
        #     reward = -reward
        # if reward<0:
        #     reward = reward*1.085
        self.totalReward+=reward
        done = next_step == len(self.df.index) - 1
        obs = self.df.iloc[next_step, :][self.features].values
        self.current_step += 1
        return obs, reward, done, {'DT':self.df.index[next_step],'totalReward':self.totalReward,'tradingCost':self.tradingCost,'holdAmt':self.holdAmt}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.net_worth = 0
        self.holdAmt=0
        self.current_step = 0
        self.totalReward = 0
        return self.df.iloc[self.current_step, :][self.features].values
        # print()
    def render(self,  close=False):
        df_plot = self.df_backup.iloc[max(0,self.current_step-1-100):self.current_step+1,:]
        # halfPos = df_plot[df_plot['holdAmt']==0.5]
        fullPos = df_plot[df_plot['holdAmt']==1]
        fullNeg= df_plot[df_plot['holdAmt']==-1]

        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(1, 1, 1)
            # self.ax = df_plot.plot()
            self.ax.clear()
            self.ax.scatter(df_plot.index, df_plot['CLOSE'], s=100)
            self.ax.scatter(fullPos.index, fullPos['CLOSE'], marker='^', s=80)
            self.ax.scatter(fullNeg.index, fullNeg['CLOSE'], marker='v', s=80)

            plt.title(f"current step process is {df_plot.index[-1].year}/{df_plot.index[-1].month}/{df_plot.index[-1].day}")

            plt.show(block=False)
            plt.pause(0.5)
        else:
            self.ax.clear()
            self.ax.plot(df_plot.index, df_plot['CLOSE'])
            # self.ax.scatter(halfPos.index,halfPos['国债10Y'],marker='v',s=50,alpha=0.5)
            self.ax.scatter(fullPos.index,fullPos['CLOSE'],marker='^',s=80)
            self.ax.scatter(fullNeg.index, fullNeg['CLOSE'], marker='v', s=80)

            # xticks rotation
            plt.xticks(rotation=45)
            plt.title(f"current step process is {df_plot.index[-1].year}/{df_plot.index[-1].month}/{df_plot.index[-1].day}")
            plt.pause(0.5)
            plt.tight_layout()
            plt.show(block=False)
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'TotalReward: {self.totalReward}')
from typing import Callable
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
def fuc_enter(x):
    return True if (x[0] <= 0 and x[1] > 0) else False

features = ['T_price:信号','buys_macd','f_RSI','CDLMORNINGDOJISTAR','CDLENGULFING','CDLDOJISTAR']
# selectedList = ['CDLMORNINGDOJISTAR', 'CDLENGULFING', 'CDLDOJISTAR', 'CDLCLOSINGMARUBOZU', 'CDLSHORTLINE','CDLLONGLINE', 'CDLSTICKSANDWICH']
patternList = ['CDLMORNINGDOJISTAR', 'CDLENGULFING','CDLDOJISTAR']

def fuc_longshort_enter(x):
    return 1 if x.values[0]==100 else -1 if x.values[0]==-100 else 0
def macd(dfData, target):
        dfData['macd_h'] = tapds.macd(dfData[target])['MACDh_12_26_9']
        # dfData['macd_x'] = tapds.macd(dfData[target])['MACDh_12_26_9']

        # dfData['cci'] = ta.cci(dfData['Close'], dfData['Close'], dfData['Close'])
        # dfData['wt'] = w_t(dfData['Close'], 12, 9)
        dfData['buys_macd'] = dfData['macd_h'].rolling(2).apply(lambda x: fuc_enter(x))
        # dfData['buys_wt'] = dfData['wt'].rolling(2).apply(lambda x: fuc_enter(x))
        # dfData['buys_wt'] = dfData['wt'].map(lambda x: True if -0.3 < x < 1.3 else False)
        # dfData['buys_final'] = dfData['buys_macd'] > 0
        return dfData
if __name__ == '__main__':
    # df = pd.read_csv('data/AAPL.csv', index_col=0)
    # df = df.sort_values('Date')
    # dataloader = DataLoader()
    # df_XY, df_yield_rate = dataloader.data_loader(10)
    df = pd.read_csv('data/AU(T+D).SGE.csv',parse_dates=['DT'],index_col=0)

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
    df[f'f_above_MA20'] = 2*((df[target] - df[f'{target}_MA20']) > 0)-1
    df[f'f_above_MA40'] = 2*((df[target] - df[f'{target}_MA40']) > 0)-1


    df[f'{target}_diff4'] = 2 * (df[target].diff(4) > 0).astype(int) - 1
    df['神奇九转'] = (df[f'{target}_diff4'].rolling(9).sum().abs() >= 9).astype(int) * df[f'{target}_diff4']
    df['f_RSI'] = talib.RSI(df[target], timeperiod=14)
    df['f_RSI'] = df['f_RSI'].map(lambda x: 0 if (x >= 10 and x <= 90) else (-1 if x < 10 else 1))
    df['f_RSI'] =  df['f_RSI'].replace(0, np.nan).ffill(limit=5).fillna(0)
    # df['f_RSI'] = 0
    df['价格空间_MAN'] = df[target].rolling(20).mean()
    df['价格空间'] = (df[target] - df['价格空间_MAN']).map(lambda x: 1 if x >= 0 else -1)
    df = macd(df, target)
    df['buys_macd'] = df['buys_macd'].replace(0, np.nan).ffill(limit=3).fillna(0)

    df['buys_macd']=df['buys_macd']
    # # df_XY.rename(columns={'T_price': '国债10Y_Price'}, inplace=True)
    #
    cl = df['CLOSE']
    op = df['OPEN']
    hi = df['HIGH']
    lo = df['LOW']
    for candle in patternList:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)
        df[candle] = df[candle].rolling(1).apply(fuc_longshort_enter).replace(0, np.nan).ffill(limit=2).fillna(0)
    #
    # df['f_above_MA10'] = df['f_above_MA10'].astype(int)
    # # features+=selectedList
    df = df[[target]+features].fillna(0)
    df['reward_pct'] = df['CLOSE'].pct_change(1).shift(-1).fillna(0)

    df['Close_diff_shft-1'] =  df[target].diff(1).shift(-1).fillna(0)
    df['Close_diff_shft-1'] = df['Close_diff_shft-1'].map(lambda x: x if x > 0 else 1.1*x)



    df['Close_diff_shft-5'] = df[target].diff(5).shift(-5).fillna(0)/5

    df_train = df[ (df.index<= pd.to_datetime("2019/12/31")) & (df.index>= pd.to_datetime("2011/12/31"))]
    coef = abs(df_train['Close_diff_shft-5'][df_train['Close_diff_shft-5'] > 0].sum() / df_train['Close_diff_shft-5'][
        df_train['Close_diff_shft-5'] < 0].sum())
    if coef < 1:
        k = 1 / coef
        df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: k * x if x > 0 else x)
    else:
        k = coef
        df['Close_diff_shft-5_modified'] = df['Close_diff_shft-5'].map(lambda x: x if x > 0 else k * x)

    df_train = df[ (df.index<= pd.to_datetime("2019/12/31")) & (df.index>= pd.to_datetime("2011/12/31"))]
    df_test = df[ (df.index> pd.to_datetime("2019/12/31"))& (df.index<= pd.to_datetime("2022/12/31"))]
    print(df_train.shape[0]+df_test.shape[0])

    # df_train['Close_diff_shft-5_modified'].cumsum().plot()
    # df_train['Close_diff_shft-5'].cumsum().plot()
    #
    # plt.legend(['Price after Adaptive Scaler','Original Price in Training', ])
    # plt.xlabel('Date', fontsize=12)
    # plt.ylabel('Price Change through Time', fontsize=12)
    # plt.tight_layout()
    #
    # plt.savefig('ExpFigs/AdaptiveScaler_g.eps', dpi=600)
    # plt.savefig('ExpFigs/AdaptiveScaler_g.png')
    # plt.show()
    # haha = df_train.groupby('T_price:信号')['Close_diff_shft-1'].agg(['mean','count'])

    env_target = 'Close_diff_shft-5_modified'
    #
    # df_train[target].plot(c='k')
    # lala  = df_train[df_train['T_price:信号']==1]
    # # plt.scatter(lala.index,lala[target],c='r',s=5)
    # # plt.show()
    # # df_train['Truth'] = 2*(df_train['Close_diff_shft-1']>0)-1
    # df_train['daily_reward'] = df_train[env_target]
    # df_train['daily_reward'].cumsum().plot(c='b')
    # plt.show()
    plot_sig =  '价格空间'
    plot_sig='buys_macd'
    plot_sig='CDLMORNINGDOJISTAR'
    plot_sig='f_RSI'
    # plot_sig='神奇九转'
    # features+=patternList
    if plot_sig in features:
        coef = 1
        f1 = np.corrcoef(df_train[plot_sig], df_train[env_target])

        df_train['daily_reward'] = df_train[env_target] * df_train[plot_sig]*coef
        df_train['daily_reward'].cumsum().plot(c='g')
        df_test['daily_reward'] = df_test[env_target] * df_test[plot_sig]*coef
        df_test['daily_reward'].cumsum().plot(c='cyan')
        df_test['Close_diff_shft-5'].cumsum().plot(c='y')
    plt.show()


    env = StockTradingEnv(df_train.copy(),env_target,features)
    # env = DummyVecEnv([lambda: env])
    policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[20,dict(pi=[20,20], vf=[20,20])])
    model = A2C('MlpPolicy', env, verbose=1, n_steps=50, gamma=0.1, policy_kwargs=policy_kwargs,
                tensorboard_log="a2c_tensorboard/")
    # # print(model.policy)
    model.learn(total_timesteps=50000,)
    # model.save(f"paperResult//gold//a2c_gold_trading2")
    # model = A2C.load(f"a2c_gold_trading2",env=env)
    # train_val = True

    # env_target = 'Close_diff_shft-1'
    # env_eval = StockTradingEnv(df_train.copy(), env_target, features=features)
    # test_model(model, env_eval, trade_obj='gold')

    env_target = 'reward_pct'
    env_eval = StockTradingEnv(df_test.copy(), env_target, features=features)
    test_model(model, env_eval,trade_obj='gold',benchmark=True)

