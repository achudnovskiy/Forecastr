#!/usr/bin/env python3

import sys
import numpy as np
from sklearn import preprocessing
import progressbar
import time
from Common.Config import Config as GlobalConfig
import Common.FileManager as FileManager
from Common.Market import Market
from .Config import Config

if GlobalConfig.USE_CUDA:
    try:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.linalg as linalg
        import skcuda.misc as misc
        useCuda = True
        linalg.init()
    except ImportError:
        useCuda = False
else:
    useCuda = False

def initModel(D, H, A):
    model = {}
    model['W1'] = np.asarray(np.random.randn(D, H) / np.sqrt(D), np.float64) # "Xavier" initialization
    model['W2'] = np.asarray(np.random.randn(H, A) / np.sqrt(H), np.float64)

    return model

def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def policy_forward(x, model):
    if len(x.shape) == 1:
        x = x[np.newaxis, ...]

    if useCuda:
        h = cuda_multiply(x.copy(), model['W1'])
        h[h<0] = 0 # ReLU nonlinearity
        logp = cuda_multiply(h, model['W2'])
    else:
        h = x.dot(model['W1'])
        h[h<0] = 0 # ReLU nonlinearity
        logp = h.dot(model['W2'])

    p = softmax(logp)

    return p, h

def cuda_multiply(mtx1, mtx2, trans1='N', trans2='N'):
    mtx1_gpu = gpuarray.to_gpu(mtx1)
    mtx2_gpu = gpuarray.to_gpu(mtx2)
    return linalg.dot(mtx1_gpu, mtx2_gpu, transa=trans1, transb=trans2).get()


def run():

    try:
        model = FileManager.restoreDataFromFile("PG")
    except:
        if GlobalConfig.FORECAST_MODE:
            print("Couldn't find the model")
            return

    market = Market()
    running_reward = None
    episode_number = 0

    t = time.time()
    

    reward_sum = 0
    done = False
    observation = market.reset()
    prev_observation = None

    progress_bar = progressbar.ProgressBar(maxval=market.currentBatchLength() - 1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    progress_bar.start()

    while done == False:
        progress_bar.update(market.iterator)
        observation_change = observation - prev_observation if prev_observation is not None else np.zeros(GlobalConfig.SLICE_SIZE * GlobalConfig.SLICE_WIDTH)
        prev_observation = observation 
        # forward the policy network and sample an action from the returned probability
        action_probability, hidden_state = policy_forward(observation_change,model)
        # roll the dice, in the softmax loss
        u = np.random.uniform()
        action_probability_cumulative = np.cumsum(action_probability)
        action = np.where(u <= action_probability_cumulative)[0][0]

        observation, reward, done, info = market.takeAction(action)
        reward_sum += reward



    progress_bar.finish()
    # print ('running mean: %f' % running_reward)
    print ('Finished in', time.time()-t, 'seconds ')
    stock, trades = market.exportDataTradingData()
    FileManager.saveTrades(stock,trades)