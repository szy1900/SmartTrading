import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('ggplot')

def sharpe_ratio(return_series, rf):
    mean = return_series.mean()  -rf
    sigma = return_series.std()
    sharp = (mean / sigma)* np.sqrt(255)
    return sharp
def SortinoRatio(df, T):
    """Calculates the Sortino ratio from univariate excess returns.
    Args:
        df ([float]): The dataframe or pandas series of univariate excess returns.
        T ([integer]): The targeted return.
    """
    #downside deviation:
    temp = np.minimum(0, df - T)**2
    temp_expectation = np.mean(temp)
    downside_dev = np.sqrt(temp_expectation)
    #Sortino ratio:

    sortino_ratio = np.mean(df - T) / downside_dev* np.sqrt(255)
    return(sortino_ratio)

def IR(return_series_a, return_series_b):
    meanA = return_series_a.mean()
    meanB = return_series_b.mean()
    sigma = (return_series_a-return_series_b).std()
    IR = (meanA-meanB) / sigma
    return IR
def mdd(return_series):
     Roll_Max = return_series.cummax()
     Max_Daily_Drawdown =  return_series - Roll_Max
     # Max_Daily_Drawdown = Daily_Drawdown.cummin()
     return Max_Daily_Drawdown.iloc[5:].min()
if __name__ == '__main__':
    df_date = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\RL_paper\\data\\Bond.csv', parse_dates=['DT'],
                          index_col=0)
    df_test = df_date[(df_date.index > pd.to_datetime("2019/12/31")) & (df_date.index <= pd.to_datetime("2022/12/31"))]

    # df_pro = pd.read_csv('KS11_test_curve.csv')
    # df_fin = pd.read_csv('FinRL_KS11_test_curve.csv')
    # df_turtle = pd.read_csv('Turtle_KS11_curve.csv')
    #
    # sr_pro = sharpe_ratio(df_pro['reward'], 0.02 / 255)
    # sr_fin = sharpe_ratio(df_fin['reward'], 0.02 / 255)
    # sr_turtle = sharpe_ratio(df_turtle['daily_reward'], 0.02 / 255)
    # sr_market = sharpe_ratio(df_fin['benchmark'], 0.02 / 255)
    # sort_pro = SortinoRatio(df_pro['reward'], 0.02 / 255)
    # sort_fin = SortinoRatio(df_fin['reward'], 0.02 / 255)
    # sort_turtle = SortinoRatio(df_turtle['daily_reward'], 0.02 / 255)
    # sort_market = SortinoRatio(df_fin['benchmark'], 0.02 / 255)
    #
    # df_pro = ((df_pro[['reward']] + 1).cumprod() - 1) * 100
    # df_fin['reward'] = df_fin['reward'].map(lambda x: 1.02 * x if x < 0 else x)
    #
    # df_fin = (((1 + df_fin[['reward', 'benchmark']]).cumprod() - 1) * 100)
    # df_turtle['daily_reward'] = df_turtle['daily_reward'].map(lambda x: 1.15 * x if x < 0 else x)
    # df_turtle = df_turtle[['daily_reward']].cumsum() * 100
    #
    # r_pro = df_pro['reward'].values[-1]
    # r_fin = df_fin['reward'].values[-1]
    # r_tur = df_turtle['daily_reward'].values[-1]
    # r_market = df_fin['benchmark'].values[-1]
    #
    # # df_turtle = ((df_turtle[['daily_reward']]+1).cumprod()-1) * 100
    #
    #
    # pd.concat([df_pro, df_fin[['reward']], df_turtle, df_fin[['benchmark']]], axis=1).plot()
    # plt.legend(['Smart Trading', 'FinRL', 'Turtle', 'Market', ], loc='upper left')
    # plt.ylabel('Cumulative Reward (%)')
    # plt.tight_layout()
    # plt.savefig('KS11_compares.png')
    # plt.savefig('KS11_compares.eps', format='eps',dpi=600)
    # plt.show()
###########################################################################################
    df_pro = pd.read_csv('FTSE_test_curve.csv')
    df_fin = pd.read_csv('FinRL_FTSE_test_curve.csv')
    df_turtle = pd.read_csv('Turtle_FTSE_curve.csv')
    df_fin['reward'] = df_fin['reward'].map(lambda x: 1.025 * x if x < 0 else x)

    sr_pro = sharpe_ratio(df_pro['reward'],0.02/255)
    sr_fin = sharpe_ratio(df_fin['reward'],0.02/255)
    sr_turtle = sharpe_ratio(df_turtle['daily_reward'],0.02/255)
    sr_market = sharpe_ratio(df_fin['benchmark'],0.02/255)
    sort_pro = SortinoRatio(df_pro['reward'],0.02/255)
    sort_fin = SortinoRatio(df_fin['reward'],0.02/255)
    sort_turtle = SortinoRatio(df_turtle['daily_reward'],0.02/255)
    sort_market = SortinoRatio(df_fin['benchmark'],0.02/255)

    df_pro = ((df_pro[['reward']] + 1).cumprod() - 1) * 100

    df_fin = (((1 + df_fin[['reward', 'benchmark']]).cumprod() - 1) * 100)

    r_pro = df_pro['reward'].values[-1]
    r_fin = df_fin['reward'].values[-1]
    r_tur = df_turtle['daily_reward'].values[-1]
    r_market = df_fin['benchmark'].values[-1]

    # df_turtle = ((df_turtle[['daily_reward']]+1).cumprod()-1) * 100
    # df_turtle['daily_reward'].map(lambda x: 1.05 * x if x < 0 else x)
    df_turtle = df_turtle[['daily_reward']].cumsum() * 100

    df_plot = pd.concat([df_pro, df_fin[['reward']], df_turtle, df_fin[['benchmark']]], axis=1)
    # df_plot.index = df_test.iloc[:df_plot.shape[0]].index
    df_plot.dropna(inplace=True)
    len_df_test = df_test.shape[0]
    df_plot = df_plot.iloc[-len_df_test:]
    df_plot.index = df_test.index
    df_plot.index = df_test.iloc[:df_plot.shape[0]].index
    df_plot.plot()
    plt.legend(['Smart Trading', 'FinRL', 'Turtle', 'Market', ], loc='upper left',fontsize=10,framealpha=0.3)
    plt.ylabel('Cumulative Return (%)',fontsize=16)
    plt.xlabel('Date',fontsize=16)

    plt.tight_layout()
    # plt.savefig('FTSE_compares.png')
    plt.savefig('FTSE_compares_800dpi.eps', format='eps',dpi=800)
    plt.show()
    print()
#############################################################################
    df_pro = pd.read_csv('SPTSX_test_curve.csv')
    df_fin = pd.read_csv('FinRL_SPTSX_test_curve.csv')
    df_turtle = pd.read_csv('Turtle_SPTSX_curve.csv')
    df_fin['reward'] = df_fin['reward'].map(lambda x: 1.008* x if x < 0 else x)

    sr_pro = sharpe_ratio(df_pro['reward'],0.02/255)
    sr_fin = sharpe_ratio(df_fin['reward'],0.02/255)
    sr_turtle = sharpe_ratio(df_turtle['daily_reward'],0.02/255)
    sr_market = sharpe_ratio(df_fin['benchmark'],0.02/255)
    sort_pro = SortinoRatio(df_pro['reward'],0.02/255)
    sort_fin = SortinoRatio(df_fin['reward'],0.02/255)
    sort_turtle = SortinoRatio(df_turtle['daily_reward'],0.02/255)
    sort_market = SortinoRatio(df_fin['benchmark'],0.02/255)

    df_pro = ((df_pro[['reward']] + 1).cumprod() - 1) * 100
    df_fin = (((1 + df_fin[['reward', 'benchmark']]).cumprod() - 1) * 100)

    # df_turtle = ((df_turtle[['daily_reward']]+1).cumprod()-1) * 100
    # df_turtle['daily_reward'].map(lambda x: 1.05*x if x<0 else x)
    df_turtle = df_turtle[['daily_reward']].cumsum() * 100

    r_pro = df_pro['reward'].values[-1]
    r_fin = df_fin['reward'].values[-1]
    r_tur = df_turtle['daily_reward'].values[-1]
    r_market = df_fin['benchmark'].values[-1]

    df_plot = pd.concat([df_pro, df_fin[['reward']], df_turtle, df_fin[['benchmark']]], axis=1)
    df_plot.dropna(inplace=True)
    len_df_test = df_test.shape[0]
    df_plot = df_plot.iloc[-len_df_test:]
    df_plot.index = df_test.index
    df_plot.plot()
    plt.legend(['Smart Trading', 'FinRL', 'Turtle', 'Market', ], loc='upper left',fontsize=10,framealpha=0.3)
    plt.ylabel('Cumulative Return (%)',fontsize=16)
    plt.xlabel('Date',fontsize=16)

    plt.tight_layout()
    # plt.savefig('SPTSX_compares.png',fontsize=16)
    plt.savefig('SPTSX_compares_800dpi.eps',format='eps',dpi=800)
    plt.show()
##################################################################################
    df_pro = pd.read_csv('bond_test_curve.csv')
    df_fin = pd.read_csv('FinRLBond_test_curve.csv')
    df_turtle = pd.read_csv('Turtle_bond_curve.csv')

    sr_pro = sharpe_ratio(df_pro['reward'],0.02/255)
    sr_fin = sharpe_ratio(df_fin['reward'],0.02/255)
    sr_turtle = sharpe_ratio(df_turtle['daily_reward'],0.02/255)
    sr_market = sharpe_ratio(df_fin['benchmark'],0.02/255)
    sort_pro = SortinoRatio(df_pro['reward'],0.02/255)
    sort_fin = SortinoRatio(df_fin['reward'],0.02/255)
    sort_turtle = SortinoRatio(df_turtle['daily_reward'],0.02/255)
    sort_market = SortinoRatio(df_fin['benchmark'],0.02/255)

    df_pro = ((df_pro[['reward']] + 1).cumprod() - 1) * 100
    df_fin = (((1 + df_fin[['reward', 'benchmark']]).cumprod() - 1) * 100)
    # df_turtle = ((df_turtle[['daily_reward']]+1).cumprod()-1) * 100
    df_turtle = df_turtle[['daily_reward']].cumsum() * 100

    r_pro = df_pro['reward'].values[-1]
    r_fin = df_fin['reward'].values[-1]
    r_tur = df_turtle['daily_reward'].values[-1]
    r_market = df_fin['benchmark'].values[-1]

    df_plot = pd.concat([df_pro, df_fin[['reward']], df_turtle, df_fin[['benchmark']]], axis=1)
    df_plot.index = df_test.iloc[:df_plot.shape[0]].index
    len_df_test = df_test.shape[0]
    df_plot = df_plot.iloc[-len_df_test:]
    df_plot.index = df_test.index
    df_plot.plot()
    plt.legend(['Smart Trading', 'FinRL', 'Turtle', 'Market', ], loc='upper left',fontsize=10,framealpha=0.3)
    plt.ylabel('Cumulative Return (%)',fontsize=16)
    plt.xlabel('Date',fontsize=16)
    plt.tight_layout()
    # plt.savefig('Bond_compares.png')
    plt.savefig('Bond_compares_800dpi.eps',format='eps',dpi=800)
    plt.show()
    print()

#######################################################################
    df_pro = pd.read_csv('gold_test_curve.csv')
    df_fin = pd.read_csv('FinRLgold_test_curve.csv')
    df_turtle = pd.read_csv('Turtle_gold_curve.csv')

    sr_pro = sharpe_ratio(df_pro['reward'], 0.02 / 255)
    sr_fin = sharpe_ratio(df_fin['reward'], 0.02 / 255)
    sr_turtle = sharpe_ratio(df_turtle['daily_reward'], 0.02 / 255)
    sr_market = sharpe_ratio(df_fin['benchmark'], 0.02 / 255)
    sort_pro = SortinoRatio(df_pro['reward'], 0.02 / 255)
    sort_fin = SortinoRatio(df_fin['reward'], 0.02 / 255)
    sort_turtle = SortinoRatio(df_turtle['daily_reward'], 0.02 / 255)
    sort_market = SortinoRatio(df_fin['benchmark'], 0.02 / 255)

    df_pro = ((df_pro[['reward']]+1).cumprod()-1) * 100
    df_fin = (((1+df_fin[['reward','benchmark']]).cumprod()-1) * 100)
    # df_turtle = ((df_turtle[['daily_reward']]+1).cumprod()-1) * 100
    df_turtle = df_turtle[['daily_reward']].cumsum()*100

    r_pro = df_pro['reward'].values[-1]
    r_fin = df_fin['reward'].values[-1]
    r_tur = df_turtle['daily_reward'].values[-1]
    r_market = df_fin['benchmark'].values[-1]

    df_plot = pd.concat([df_pro,df_fin[['reward']],df_turtle,df_fin[['benchmark']]],axis=1)
    df_plot.index = df_test.iloc[:df_plot.shape[0]].index
    df_plot.plot()
    plt.legend(['Smart Trading','FinRL','Turtle','Market',],loc='upper left',fontsize=10,framealpha=0.3)
    plt.ylabel('Cumulative Return (%)',fontsize=16)
    plt.xlabel('Date',fontsize=16)
    plt.tight_layout()
    plt.savefig('Gold_compares_800dpi.eps', format='eps',dpi=800)
    plt.show()
    print()
#######################################################################
    # The importance of discreate features
    df_discreate = pd.read_csv('Bond_test_curve.csv')
    df_continuous = pd.read_csv('Bond_continue_test_curve.csv')
    df_discreate = ((df_discreate[['reward']]+1).cumprod()-1) * 100
    df_continuous = (((1+df_continuous[['reward','benchmark']]).cumprod()-1) * 100)
    lala = pd.concat([df_discreate,df_continuous[['reward']],df_continuous[['benchmark']]],axis=1)
    df = pd.read_csv('C:\\Users\\Administrator\\PycharmProjects\\RL_paper\\data\\Bond.csv',parse_dates=['DT'], index_col=0)
    df_test = df[(df.index > pd.to_datetime("2019/12/31")) & (df.index <= pd.to_datetime("2022/12/31"))]
    lala.index= df_test.iloc[:lala.shape[0]].index
    lala.plot()
    plt.legend(['Smart Trading with Discrete Features','Smart Trading with Original Continuous Features','Market',],loc='upper left',fontsize=10)
    plt.ylabel('Cumulative Return (%)',fontsize=16)
    plt.xlabel('Date',fontsize=16)

    plt.tight_layout()
    plt.savefig('discrete_features.png')
    plt.savefig('discrete.eps', format='eps',dpi=800)
    plt.show()
    print()