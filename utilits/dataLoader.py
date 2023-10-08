import pandas as pd
from scipy.stats import rankdata
import talib as ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_ta as tapds


def signal_scope(sig_name_list, df_X, df_X_prices, flag=False):
    # 针对0-1信号
    # df_X_prices = df_X[['treasury_bond_rate_10y']]
    fig = plt.figure(figsize=(15, 8))
    if flag:
        ax = sns.lineplot(x=df_X.index, y=df_X['f_EMA10'], label="Data", color='k', linewidth=1,
                          alpha=0.4)
        ax = sns.lineplot(x=df_X.index, y=df_X['f_EMA15'], label="Data", color='k', linewidth=1,
                          alpha=0.4)
        ax = sns.lineplot(x=df_X.index, y=df_X['f_EMA20'], label="Data", color='k', linewidth=1,
                          alpha=0.4)
    for sig_name in sig_name_list:

        ax = sns.lineplot(x=df_X.index, y=df_X_prices.iloc[:, -1], label="Data", color='k', linewidth=1,
                          alpha=0.8)
        if sig_name != '流动性弹性信号' and sig_name != '风险偏好弹性' and sig_name != '基本面弹性':
            df_sig_interest = df_X[[sig_name]]
            ax = sns.scatterplot(x=df_sig_interest.index,
                                 y=df_X_prices.loc[df_sig_interest.index, :].values.flatten().tolist(),
                                 hue=df_sig_interest[sig_name].values, palette='vlag')
        else:
            df_sig_interest = df_X[[sig_name]][df_X[sig_name] != 0]
            ax = sns.scatterplot(x=df_sig_interest.index,
                                 y=df_X_prices.loc[df_sig_interest.index, :].values.flatten().tolist(),
                                 hue=df_sig_interest[sig_name].values, palette='vlag')
        plt.title(sig_name, fontsize=10)
        plt.show()
        print()

def plotSigs(dfData, N, enterSig, target):
    plt.figure(figsize=(12,9))
    dfPlot = dfData.iloc[-N:, :]
    ax = dfPlot[target].plot()
    # if lineSig is not None:
    #     dfPlot[lineSig].plot(ax=ax)
    dfPlot.dropna(inplace=True)
    dfbuys = dfPlot[dfPlot[enterSig] == 1]
    ax.scatter(x=dfbuys.index, y=dfbuys[target], marker='v', color='g')

    dfsell = dfPlot[dfPlot[enterSig] == -1]
    ax.scatter(x=dfsell.index, y=dfsell[target], marker='^', color='r')
    # for idx,row in dfPlot.iterrows():
    #     ax.axvline(x =idx,color = row['color'],alpha=0.1 )
    plt.show()

class DataLoader():
    def __init__(self):
        pass

    def cal_multitrend_signal(self,df, colname):
        df[f'{colname}MA9'] = df[f'{colname}'].rolling(9).mean()
        df[f"{colname}MA9_shift"] = df[f'{colname}MA9'].shift(1)
        df[f'{colname}MA10'] = (df[f'{colname}'] * 3 + df[f"{colname}MA9_shift"] * 9) / 12
        # df[f'{colname}MA10'] = df[f'{colname}'].rolling(10).mean()
        # df[f'{colname}MA15'] = df[f'{colname}'].rolling(15).mean()
        # df[f'{colname}MA20'] = df[f'{colname}'].rolling(20).mean()
        # df[f'{colname}MA10'] = df[f'{colname}'].rolling(10).mean()
        df[f'{colname}MA15'] = df[f'{colname}'].rolling(15).mean()
        df[f'{colname}MA20'] = df[f'{colname}'].rolling(20).mean()
        def cal_signal(x):
            if x[f'{colname}MA10'] >= x[f'{colname}MA15'] and x[f'{colname}MA15'] >= x[f'{colname}MA20']:
                return 1 if colname == "T_price" else -1
            elif x[f'{colname}MA10'] <= x[f'{colname}MA15'] and x[f'{colname}MA15'] <= x[f'{colname}MA20']:
                return -1 if colname == "T_price" else 1
            else:
                return 0

        df[f'{colname}:信号'] = df.apply(lambda x: cal_signal(x), axis=1)
        # df = df.drop([f'{colname}MA9',f'{colname}MA9_shift',f'{colname}MA10',f'{colname}MA15',f'{colname}MA20'],axis=1)
        return df

    def delta(self,df, period=1):
        """
        Wrapper function to estimate difference.
        :param df: a pandas DataFrame.
        :param period: the difference grade.
        :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
        """
        return df.diff(period)

    def rolling_rank(self,na):
        """
        Auxiliary function to be used in pd.rolling_apply
        :param na: numpy array.
        :return: The rank of the last value in the array.
        """
        return rankdata(na,method='min')[-1]

    def ts_rank(self,df, window=10):
        """
        Wrapper function to estimate rolling rank.
        :param df: a pandas DataFrame.
        :param window: the rolling window.
        :return: a pandas DataFrame with the time-series rank over the past window days.
        """
        return df.rolling(window).apply(self.rolling_rank)

    def compute_technical_indicators(self,df):
        df['f_alpha007'] = self.ts_rank(abs(self.delta(df[['国债10Y']], 5)), 7) * np.sign(self.delta(df[['国债10Y']], 7))
        df['f_alpha004'] = self.ts_rank(df[['国债10Y']], 10)
        df['f_RSI'] = ta.RSI(df['国债10Y'], timeperiod=14)
        return df
    def data_loader(self,diffN=10):
        df_XY = pd.read_csv('data/模型信号_沈博专属.csv', encoding='utf-8',parse_dates=['DT']).set_index('DT')
        # df_XY['FR007'] = df_lidiquity.loc[df_XY.index, 'FR007']
        # df_XY['DR007'] = df_lidiquity.loc[df_XY.index, 'DR007']
        # df_XY['REPO1Y'] = df_lidiquity.loc[df_XY.index, 'REPO1Y']
        # df_XY['REPO1Y_MA40'] = df_lidiquity.loc[df_XY.index, 'REPO1Y_MA40']
        # df_XY['REPO1Y空间'] = 2*(df_XY['REPO1Y']<df_XY['REPO1Y_MA40']).astype(int)-1
        #
        # df_XY['3MSHIBOR1Y'] = df_lidiquity.loc[df_XY.index, '3MSHIBOR1Y']

        df_XY[f'国债10Y_diff{diffN}'] = df_XY['国债10Y'].diff(diffN).shift(-diffN)
        df_XY = self.compute_technical_indicators(df_XY)
        # df_XY = pd.concat([df_XY, df_indicator], axis=1, join='outer')
        df_XY = self.cal_multitrend_signal(df_XY, 'T_price')
        # df_XY = self.cal_multitrend_signal(df_XY, '流动性')
        df_XY = self.cal_multitrend_signal(df_XY, '国债10Y')
        # df_XY = self.cal_multitrend_signal(df_XY, 'REPO1Y')

        # df_XY = self.cal_multitrend_signal(df_XY, '基本面')
        # df_XY = self.cal_multitrend_signal(df_XY, '风险偏好')
        df_XY['国债10Y_MA20'] = ta.MA( df_XY['国债10Y'].values, timeperiod=20)
        df_XY['国债10Y_MA15'] = ta.MA( df_XY['国债10Y'].values, timeperiod=15)
        df_XY['国债10Y_MA10'] = ta.MA( df_XY['国债10Y'].values, timeperiod=10)

        df_XY['国债10Y_MA11'] = ta.MA( df_XY['国债10Y'].values, timeperiod=11)
        df_XY['国债10Y_MA9'] = ta.MA( df_XY['国债10Y'].values, timeperiod=9)
        df_XY['国债10Y_MA8'] = ta.MA( df_XY['国债10Y'].values, timeperiod=8)

        df_XY['国债10Y_MA5'] = ta.MA( df_XY['国债10Y'].values, timeperiod=5)

        df_XY['f_MA20_dir'] = (df_XY['国债10Y_MA20'] - df_XY['国债10Y_MA20'].shift(1))>0
        df_XY['f_above_MA20'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA20'])>0
        # df_XY['f_cross_MA20'] =  df_XY['f_above_MA20'] -  df_XY['f_above_MA20'].shift(1)

        df_XY['f_above_MA5'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA5'])>0
        df_XY['f_above_MA8'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA8'])>0
        df_XY['f_above_MA9'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA9'])>0
        df_XY['f_above_MA10'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA10'])>0
        df_XY['f_above_MA11'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA11'])>0
        df_XY['f_above_MA15'] = (df_XY['国债10Y'] - df_XY['国债10Y_MA15'])>0

        # df_XY['f_cross_MA10'] =  df_XY['f_above_MA10'] -  df_XY['f_above_MA10'].shift(1)

        # df_XY['f_cross_MA5'] =  df_XY['f_above_MA5'] -  df_XY['f_above_MA5'].shift(1)

        df_XY['f_above_MAShort'] =  df_XY['f_above_MA10']|df_XY['f_above_MA9']

        df_XY['流动性MA10'] = ta.MA(df_XY['流动性新'].values, timeperiod=10)
        df_XY['流动性MA15'] = ta.MA(df_XY['流动性新'].values, timeperiod=15)
        df_XY['流动性MA20'] = ta.MA(df_XY['流动性新'].values, timeperiod=20)

        df_XY['基本面MA10'] = ta.MA(df_XY['基本面'].values, timeperiod=10)
        df_XY['基本面MA15'] = ta.MA(df_XY['基本面'].values, timeperiod=15)
        df_XY['基本面MA20'] = ta.MA(df_XY['基本面'].values, timeperiod=20)

        df_XY['流动性信号'] = df_XY.apply(
            lambda x: 1 if x['流动性MA10'] < x['流动性MA15'] < x['流动性MA20'] else -1 if x['流动性MA10'] > x['流动性MA15'] > x[
                '流动性MA20'] else 0, axis=1)
        df_XY['基本面信号'] = df_XY.apply(
            lambda x: 1 if x['基本面MA10'] < x['基本面MA15'] < x['基本面MA20'] else -1 if x['基本面MA10'] > x['基本面MA15'] > x[
                '基本面MA20'] else 0, axis=1)
        df_XY['胜率自算'] = df_XY.apply(
            lambda x: 1 if (x['流动性信号'] + x['基本面信号']) > 0 else -1 if (x['流动性信号'] + x['基本面信号']) < 0 else 0, axis=1)
        # temp = df_XY[['胜率自算','胜率']].copy().dropna()
        # (temp['胜率自算'] == temp['胜率']*-1).sum()/len(temp['胜率自算'])

        df_XY['价格空间_MAN'] = df_XY['T_price'].rolling(20).mean()
        df_XY['价格空间'] = (df_XY['T_price'] - df_XY['价格空间_MAN']).map(lambda x: 1 if x >= 0 else -1)

        df_XY['牛熊震荡'] = df_XY['价格空间'] + df_XY['基本面空间']

        df_XY['牛熊震荡备忘'] = 1
        for idx, row in df_XY.reset_index().iterrows():
            if (idx>= 1) and (idx <= df_XY.shape[0] - 1):
                prev_row = df_XY.iloc[idx - 1]
                next_future_day = df_XY.reset_index().iloc[min(df_XY.shape[0]-1,idx+3)]
                if row['牛熊震荡'] * prev_row['牛熊震荡'] == -1:
                    df_XY.loc[row['DT']:next_future_day['DT'], '牛熊震荡备忘'] = 0
        df_XY['牛熊震荡_平滑'] = df_XY['牛熊震荡'] * df_XY['牛熊震荡备忘']
        df_XY['牛熊震荡_平滑'] = df_XY['牛熊震荡_平滑'].map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        df_XY['国债10Y_diff4'] = 2*(df_XY['国债10Y'].diff(4)>0).astype(int)-1
        df_XY['神奇九转'] = (df_XY['国债10Y_diff4'].rolling(9).sum().abs()>=9).astype(int)*df_XY['国债10Y_diff4']
        df_XY['f_RSI'] = df_XY['f_RSI'].map(lambda x: 0 if (x >= 18 and x <= 84) else (-1 if x < 20 else 1))
        df_XY['反转信号'] = df_XY['反转信号'].map(lambda x: 0 if (x >= -4 and x <= 4) else -1 if x < -4 else 1)

        def fuc_enter(x):
            return 1 if (x[0] > 0 and x[1] < 0) else -1 if (x[0] < 0 and x[1] > 0) else 0
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
        df_XY = macd(df_XY, 'T_price')
        # plotSigs(df_XY, 600, 'buys_macd', 'T_price')
        print()
        # df_XY['10年国债期货:MA40'] = ta.MA(df_XY['10年国债期货'].values,timeperiod = 40)
        # df_XY['10年国债期货:MA40'] = df_XY.apply(lambda x: 1 if x['10年国债期货'] > x['10年国债期货:MA40'] else -1 if x['10年国债期货'] < x['10年国债期货:MA40'] else 0,axis =1)
        # df_XY['国债10Y:MA40'] = ta.MA(df_XY['国债10Y'].values,timeperiod = 40)
        # df_XY['国债10Y:MA40'] = df_XY.apply(lambda x: -1 if x['国债10Y'] > x['国债10Y:MA40'] else 1 if x['国债10Y'] < x['国债10Y:MA40'] else 0,axis =1)
        # df_XY['国债price_sum'] = df_XY['国债10Y:MA40']+df_XY['10年国债期货:MA40']+df_XY['价格空间']
        # df_XY['国债price_sum'] = df_XY.apply(
        #     lambda x: -1 if x['国债price_sum'] <0 else 1 if x['国债price_sum'] >0 else 0, axis=1)
        # df_s1 = df_XY[['国债price_sum', '国债10Y_diff10']].groupby(["国债price_sum"]).agg(['mean', 'count'])
        # df_s2 = df_XY[['国债10Y:MA40', '国债10Y_diff10']].groupby(["国债10Y:MA40"]).agg(['mean', 'count'])
        # df_s3 = df_XY[['10年国债期货:MA40', '国债10Y_diff10']].groupby(["10年国债期货:MA40"]).agg(['mean', 'count'])
        # df_s4 = df_XY[['价格空间', '国债10Y_diff10']].groupby(["价格空间"]).agg(['mean', 'count'])


        # df_XY = df_XY.dropna(subset = [])
        # plt.figure(figsize=(16, 6))
        # for date in df_XY.index:
        #     print(date)
        #     print(df_XY.loc[date, '价格空间'])
        #     if df_XY.loc[date, '价格空间'] >= 1:
        #         plt.axvline(x=date, color=(1, 0, 0, 0.2), linestyle='-')
        #     elif df_XY.loc[date, '价格空间'] <= -1:
        #         plt.axvline(x=date, color=(0, 1, 0, 0.2), linestyle='-')
        #     # else:
        #     #     plt.axvline(x=date, color=(1, 1, 0, 0.2), linestyle='-')
        # plt.plot(df_XY.index, df_XY['国债10Y'].values,
        #          marker='', linestyle='-', linewidth=3, color='k', alpha=0.8)
        # plt.show()
        # df_XY['国债10Y'].rolling(20).mean()
        # label_col = 'label'
        df_yield_rate = df_XY[['国债10Y']]
        # df_XY.drop(columns=['国债10Y'], inplace=True)
        return df_XY, df_yield_rate
