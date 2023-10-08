import matplotlib.pyplot as plt
import pandas as pd
def test_model(model, env_eval, trade_obj='gold',benchmark=False):
    # env_eval = StockTradingEnv(df_train.copy(), env_target, features=features)
    obs = env_eval.reset()
    # env_eval.render()
    all_obs = []
    records = []

    # for idx in range(env_eval.datalength - 1):  ##多少步
    #
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs_ori = obs.copy()
    #     obs, rewards, dones, info = env_eval.step(action)
    #     all_obs.append(tuple(obs))
    #     records.append({'idx': idx, 'DT': info['DT'], 'obs': tuple(obs_ori), 'action': action, 'nex_obs': tuple(obs),
    #                     'holdAmt': info['holdAmt'], 'totalReward': info['totalReward'],
    #                     'tradingCost': info['tradingCost']})
    #     print(info['totalReward'])
    #
    # lala = pd.DataFrame(records).set_index('DT')
    # lala['totalReward'].plot()
    # plt.title('A2C训练集累计收益')
    #
    # # env_eval = StockTradingEnv(df_test.copy(), env_target, features=features)
    # obs = env_eval.reset()
    # # env_eval.render()
    # all_obs = []
    records_test = []
    for _ in range(env_eval.datalength - 1):  ##多少步
        # obs1 = np.vstack(obs).astype(np.float)

        action, _states = model.predict(obs, deterministic=True)
        obs_ori = obs.copy()
        obs, rewards, dones, info = env_eval.step(action)
        all_obs.append(tuple(obs))
        records_test.append({'DT': info['DT'], 'obs': tuple(obs_ori), 'action': action, 'nex_obs': tuple(obs),
                             'holdAmt': info['holdAmt'], 'reward':rewards,'totalReward': info['totalReward'],
                             'tradingCost': info['tradingCost']})

        # records_test.append({'DT': info['DT'], 'totalReward': info['totalReward'], 'holdAmt': info['holdAmt'],'obs':tuple(obs),'action':action})


    lala_test = pd.DataFrame(records_test).set_index('DT')
    # lala_test.loc[:,['reward','totalReward']].cumprod().plot()
    # lala_test['totalReward'].plot(label='模型累计收益')
    if benchmark:
        obs = env_eval.reset()
        for idx in range(env_eval.datalength - 1):  #
            obs, rewards, dones, info = env_eval.step(1)
            lala_test.loc[lala_test.index[idx], 'benchmark'] = rewards
        # lala_test['totalReward'].plot(label='模型累计收益')
        # lala_test['benchmark'].cumsum().plot(label='基准累计收益')
        (((1+lala_test[['reward']]).cumprod()-1)*100).plot(label=['模型'])
        value = (((1+lala_test[['reward']]).cumprod()-1)*100)
        print(f"totalReward={value['reward'][-1] }%")
        lala_test.to_csv(f'{trade_obj}_test_curve.csv', index=True, encoding='utf-8-sig')
        plt.legend()
        plt.ylabel('Total Return(%)')
        plt.title('A2C Total Return in test dataset')
        plt.show()
    else:
        haha = lala_test
        haha['totalReward'].plot(label='totalReward')
        # (1+haha[['reward']]).cumprod().plot(label=['模型'])

        # haha = haha.groupby(['obs']).agg({'action': 'mean'}).reset_index()
        # haha.to_csv(f'paperResult\\{trade_obj}\\state_action.csv', index=False, encoding='utf-8-sig')
        # # haha.to_csv(f'paperResult\\{trade_obj}\\records.csv', index=True, encoding='utf-8-sig')
        # haha.to_csv(f'ExpFigs\\compare\\{trade_obj}_train_curve.csv', index=True, encoding='utf-8-sig')
        #
        # print('save!')
        plt.legend(['totalReward'])
        plt.title('totalReward in test dataset')
        plt.ylabel('totalReward')

        plt.show()

