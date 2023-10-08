import matplotlib.pyplot as plt
import seaborn as sns

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
            df_sig_interest = df_X[[sig_name]]
            ax = sns.scatterplot(x=df_sig_interest.index,
                                 y=df_X_prices.loc[df_sig_interest.index, :].values.flatten().tolist(),
                                 hue=df_sig_interest[sig_name].values, palette='vlag')
        plt.title(sig_name, fontsize=10)
        plt.show()
        print()