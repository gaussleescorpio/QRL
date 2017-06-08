import pandas as pd
import os
import numpy as np
np.random.seed(3221)
pd.set_option("display.max_columns", 500)
from sklearn import linear_model
from matplotlib import pylab as plt
import numba
from backtesting import Backtest, sharpe, odd_sharpe
from sys import platform
from sklearn.preprocessing import StandardScaler

SIM_DATA_LEN = 2000
MAX_HOLDINGS = 1
holdings = 0

if platform == "linux":
    data = pd.read_csv("/home/gauss/Downloads/fullorderbook.csv")
if platform == "darwin":
    data = pd.read_csv("/Users/gausslee/Downloads/fullorderbook-2.csv")

s = StandardScaler()
scaler = s.fit(data[0:50000][["b1", "a1", "bs1", "as1"]])
data = data.iloc[60000:65000+SIM_DATA_LEN].reset_index()


def load_data(data, features=[]):
    return data[features]


def scale_data(data):
    s = StandardScaler()
    tr_data = s.fit_transform(data)
    return tr_data



def cut_data_to_init_states(data, cut_size=10, state_features=[], flatten=True):
    global scaler
    res_data = None
    ref_data = data[state_features]
    ref_data = scaler.transform(ref_data)
    # overlapping every time 9 cols
    overlapping_size = cut_size - 1
    new_colum_size = (ref_data.shape[0] - cut_size) // (cut_size - overlapping_size) + 1
    new_width = ref_data.shape[1]
    isize = ref_data.itemsize
    res_data = np.lib.stride_tricks.as_strided(ref_data.copy("C"),
                                               shape=(new_colum_size, cut_size, new_width),
                                               strides=(new_width*isize, new_width*isize,
                                                          isize))
    if flatten:
        res_data = res_data.reshape(res_data.shape[0], res_data.shape[1]*res_data.shape[2])
    return res_data[0,:], res_data

init_states, new_data = cut_data_to_init_states(data,
                                                cut_size=20,
                                                state_features=["b1", "a1", "bs1", "as1"],
                                                flatten=False)


def comb_current_state(data_states, prev_action):
    return data_states


def actor(qval, epsilon=0.1):
    if (np.random.random() < epsilon):  # maybe choose random action if not the last epoch
        action = np.random.randint(0, 3)  # assumes 3 different actions
    else:  # choose best action from Q(s,a) values
        action = (np.argmax(qval))
    return action


def take_action(action, data_states, trading_signals, time_step, window=10):
    global holdings
    # Jump to next timestep as next state
    time_step += window + 1
    if time_step == data_states.shape[0]:
        terminate_state = 1
        next_state = comb_current_state(data_states[-1, :], action)
        trading_signals.loc[time_step-1] = 0
        trading_signals.loc[time_step] = 0
        return next_state, time_step, trading_signals, terminate_state

    next_state = comb_current_state(data_states[time_step, :], action)
    if action == 0:
        trading_signals.loc[time_step-1] = 0
    if action == 1:
        if holdings < 0:
            trading_signals.loc[time_step-1] = -holdings
            holdings = 0
        elif abs(holdings) < MAX_HOLDINGS:
            holdings += 1
            trading_signals.loc[time_step-1] = 1
        else:
            trading_signals.loc[time_step-1] = 0
    if action == 2:
        if holdings > 0:
            trading_signals.loc[time_step-1] = -holdings
            holdings = 0
        elif abs(holdings) < MAX_HOLDINGS:
            holdings -= 1
            trading_signals.loc[time_step-1] = -1
        else:
            trading_signals.loc[time_step-1] = 0
    terminate_state = 0
    return next_state, time_step, trading_signals, terminate_state


def get_reward(new_state, time_step, action, price_data, trading_signals, terminal_state, epoch,window=10):
    reward = 0.0
    # if trading_signals.iloc[time_step] < 0:
    #     buy_pos = trading_signals[trading_signals > 0]
    #     if not buy_pos.empty:
    #         buy_pos_ind = buy_pos.index[-1]

    bt = Backtest(price=price_data.iloc[0:time_step+1],
                  signal=trading_signals.iloc[0:time_step],
                  signalType="shares")
    if not bt.data.empty:
        reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2]) * bt.data['shares'].iloc[-1])
        # reward = bt.data['pnl'].iloc[-1] - bt.data['pnl'].iloc[-2]
        # if time_step > 100:
        #     reward = odd_sharpe(bt.data["pnl"].iloc[time_step-100:time_step]) - \
        #              odd_sharpe(bt.data["pnl"].iloc[time_step-100:time_step-1])
        #     reward *= 10
        # else:
        #     reward = odd_sharpe(bt.data["pnl"].iloc[0:time_step]) - \
        #              odd_sharpe(bt.data["pnl"].iloc[0:time_step-1])
        #     reward *= 10
    if terminal_state == 1:
        #save a figure of the test set
        bt = Backtest(price_data, trading_signals, signalType='shares')
        reward = odd_sharpe(bt.data["pnl"])#bt.pnl.iloc[-1]
        plt.figure(figsize=(3, 4))
        bt.plotTrades()
        plt.axvline(x=400, color='black', linestyle='--')
        plt.text(250, 400, 'training data')
        plt.text(450, 400, 'test data')
        plt.suptitle(str(epoch) + "reward %f" % reward )
        plt.show()
    return reward


import tflearn
import tensorflow as tf

window = 20
input_size = window * 4 + 1

tf.reset_default_graph()
input_data = tflearn.input_data(shape=[None, window, new_data.shape[-1]])
net1 = tflearn.layers.lstm(incoming=input_data,
                           n_units=50, activation="Leaky_Relu")
net1 = tflearn.dropout(net1, 0.6)
output = tflearn.layers.fully_connected(net1, n_units=3, activation="linear")
model_config = tflearn.regression(incoming=output, loss="mean_square")
model = tflearn.DNN(output, tensorboard_verbose=0)
#model.load("/Users/gausslee/Documents/programming/jupytercodes/RL_model/updatedmodel")
model.load("plt/updatedmodel_8")

# print(model.get_weights(net1.W))

import time
time.sleep(3)
from collections import namedtuple
replay_memory = []
replay_memory_size = 10 * SIM_DATA_LEN
record = namedtuple("record", ["states", "actions", "update", "next_states"])
epoches = 100
epsilon = 0.1
trading_signals = pd.Series(index=np.arange(len(new_data) + window ))

status = 1
terminate_state = 0
start_states = comb_current_state(init_states, 0)
time_step = 0
total_reward = 0
gamma = 0.01
ii = 0
print("starting epoch %i" % ii)

while(status == 1):
    if (time_step < window):
        trading_signals.loc[time_step] = 0
        time_step += 1
        continue
    qval = model.predict(start_states.reshape([-1, window, new_data.shape[-1]]))
    action = actor(qval, 0.0)
    print("action: %i" % action)
    print(qval)
    next_state, time_step, trading_signals, terminate_state = take_action(action=action, data_states=new_data,
                                                                          trading_signals=trading_signals,
                                                                          time_step=time_step - window, window=window)
    print("trading_signals: %i" % trading_signals.iloc[time_step - 1])
    reward = get_reward(next_state, time_step=time_step,
                        action=action, price_data=data["price"], trading_signals=trading_signals,
                        terminal_state=terminate_state, epoch=ii, window=window)
    total_reward += reward
    time.sleep(0.01)
    start_states = next_state
    print(time_step)
    if terminate_state == 1:
        epsilon -= 1.0 / (epoches*10)
        print("final reward %f for episode %i" % (reward, ii))
        status = 0
print("total reward %f" % total_reward)
