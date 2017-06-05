import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
import os
import numpy as np
pd.set_option("display.max_columns", 500)
from sklearn import linear_model
from matplotlib import pylab as plt
import numba

data = pd.read_csv("/home/gauss/Downloads/fullorderbook.csv")


def load_data(data, features=[]):
    return data[features]


def cut_data_to_init_states(data, cut_size = 10, state_features=[]):
    res_data = None
    ref_data = data[state_features]
    # overlapping every time 9 cols
    overlapping_size = 9
    new_colum_size = (ref_data.shape[0] - cut_size) // (cut_size - overlapping_size) + 1
    new_width = ref_data.shape[1]
    isize = ref_data.values.itemsize
    res_data = np.lib.stride_tricks.as_strided(ref_data.values.copy("C"),
                                               shape=(new_colum_size, cut_size, new_width),
                                               strides = (new_width*isize, new_width*isize,
                                                         isize))
    res_data = res_data.reshape(res_data.shape[0], res_data.shape[1]*res_data.shape[2])
    return res_data[0,:], res_data

init_states, new_data = cut_data_to_init_states(data,
                                                cut_size=10,
                                                state_features=["b1", "a1", "bs1", "as1"])


def comb_current_state(data_states, prev_action):
    return np.hstack((data_states, prev_action))


def actor(qval, epsilon=0.1):
    if (np.random.random() < epsilon):  # maybe choose random action if not the last epoch
        action = np.random.randint(0, 3)  # assumes 4 different actions
    else:  # choose best action from Q(s,a) values
        action = (np.argmax(qval))
    return action


def take_action(action, data_states, trading_signals, time_step, window=10):
    if time_step + window == data_states.shape[0]:
        terminate_state = 1
        next_state = comb_current_state(data_states[-1, :], action)
        trading_signals.loc[time_step + window] = 0
        return next_state, time_step + window, trading_signals, terminate_state

    next_state = comb_current_state(data_states[time_step, :], action)
    if action == 0:
        trading_signals.loc[time_step + window] = 0
    if action == 1:
        trading_signals.loc[time_step + window] = 100
    if action == 2:
        trading_signals.loc[time_step + window] = -100
    terminate_state = 0
    time_step += window + 1
    return next_state, time_step, trading_signals, terminate_state


def reward_generator(new_state, time_step, action, price_data, trading_signals, terminal_state, window=10):
    reward = 0
    if terminal_state == 0:
        if trading_signals[time_step + window] != trading_signals[time_step + window - 1]:
            i = 1
            while trading_signals[time_step + window - i] == trading_signals[time_step + window - i - 1] and \
                                                    time_step + window - i - 1 > 0:
                i += 1
            if trading_signals[time_step] > 0:
                # print(reward, price_data.loc[time_step + window, "a1"], price_data.loc[time_step - i + window, "b1"])
                reward = (price_data.loc[time_step + window, "a1"] - price_data.loc[time_step - i + window, "b1"]) \
                         * trading_signals[time_step + window] * -1
            else:
                reward = (price_data.loc[time_step + window, "b1"] - price_data.loc[time_step - 1 + window, "a1"]) \
                         * trading_signals[time_step + window] * -1
    return reward




import tensorflow as tf
import tflearn

window = 10
input_size = window * 4 + 1

trading_signals = pd.Series(index=np.arange(len(new_data) + window ))



tf.reset_default_graph()
input_net = tflearn.input_data((None, input_size))
net1 = tflearn.fully_connected(input_net, 400, activation="ReLU")
output = tflearn.fully_connected(net1, 3, activation="linear")

model_config = tflearn.regression(output, learning_rate=0.01,  batch_size=1)

model = tflearn.DNN(model_config, tensorboard_verbose=3,
                    tensorboard_dir="/tmp/tflearn/")


#model.load("/Users/gausslee/Documents/programming/jupytercodes/RL_model/updatedmodel")
model.load("updatedmodel")

import time
status = 1
terminate_state = 0
start_states = comb_current_state(init_states, 0)
time_step = 0
total_reward = 0
gamma = 0.1
while (status == 1):
    if (time_step < 10):
        trading_signals.loc[time_step] = 0
        time_step += 1
        continue
    qval = model.predict(start_states.reshape(1, -1))
    action = np.argmax(qval)
    next_state, time_step, trading_signals, terminate_state = take_action(action=action, data_states=new_data,
                                                                          trading_signals=trading_signals,
                                                                          time_step=time_step - 10)
    reward = reward_generator(next_state, time_step=time_step - 1 - 10,
                              action=action, price_data=data, trading_signals=trading_signals,
                              terminal_state=terminate_state, window=10)
    total_reward += reward
    time.sleep(0.01)
    start_state = next_state
    if time_step % 10000 == 1:
        print(float(time_step)/trading_signals.shape[0])
    if terminate_state == 1:
        status = 0
print("reward %f" % total_reward)


from matplotlib import pylab as plt
fig1, ax1 = plt.subplots(1, 1)
y1 = data["a1"]
y2 = data["b1"]
ax1.plot(range(y1.shape[0]), y1)
ax1.plot(range(y2.shape[0]), y2, color="y")
temp = trading_signals.copy()
temp[trading_signals != 100] = np.nan
temp[trading_signals == 100] = 1.0
by1 = y1*temp.iloc[0:-2]
temp = trading_signals.copy()
temp[trading_signals != -100] = np.nan
temp[trading_signals == -100] = 1.0
sy1 = y2*temp.iloc[0:-2]
ax1.plot(range(y1.shape[0]), by1, "g*")
ax1.plot(range(y2.shape[0]), sy1, "ro")
plt.show()
