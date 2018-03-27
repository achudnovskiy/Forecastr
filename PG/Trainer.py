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

def policy_backward(eph, epx, epdlogp, model):
    #""" backward pass. (eph is array of intermediate hidden states) """

    if useCuda:
        dW2 = cuda_multiply(eph, epdlogp, 'T')
        dh = cuda_multiply(epdlogp, model['W2'], 'N', 'T')
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = cuda_multiply(epx, dh, 'T')
    else:
        dW2 = eph.T.dot(epdlogp)
        dh = epdlogp.dot(model['W2'].T)
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = epx.T.dot(dh)

    return {'W1':dW1, 'W2':dW2}

def discount_rewards(r, gamma):
    #""" take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def run():

    try:
        model = FileManager.restoreDataFromFile("PG")
    except:
        model = initModel(GlobalConfig.SLICE_SIZE * GlobalConfig.SLICE_WIDTH, Config.HIDDEN_LAYER_COUNT,  GlobalConfig.ACTIONS)

    episode_count = GlobalConfig.EPISODE_COUNT
    market = Market()
    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items() } # rmsprop memory
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    episode_number = 0

    t = time.time()
    
    progress_bar = progressbar.ProgressBar(maxval=episode_count - 1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    progress_bar.start()
    while episode_number < episode_count:

        if GlobalConfig.FORECAST_MODE == False:
            progress_bar.update(episode_number)
        reward_sum = 0
        done = False
        observation = market.reset()
        prev_observation = None
        while done == False:
            observation_change = observation - prev_observation if prev_observation is not None else np.zeros(GlobalConfig.SLICE_SIZE * GlobalConfig.SLICE_WIDTH)
            prev_observation = observation 
            # forward the policy network and sample an action from the returned probability
            action_probability, hidden_state = policy_forward(observation_change,model)
            # roll the dice, in the softmax loss
            u = np.random.uniform()
            action_probability_cumulative = np.cumsum(action_probability)
            action = np.where(u <= action_probability_cumulative)[0][0]

            # record various intermediates (needed later for backprop)
            xs.append(observation_change) # observation
            hs.append(hidden_state) # hidden state

            #softmax loss gradient
            dlogsoftmax = action_probability.copy()
            dlogsoftmax[0,action] -= 1 #-discouted reward 
            dlogps.append(dlogsoftmax)

            observation, reward, done, info = market.takeAction(action)
            reward_sum += reward
            drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.asarray(np.vstack(xs), np.float64)
        eph = np.asarray(np.vstack(hs), np.float64)

        epdlogp = np.asarray(np.vstack(dlogps), np.float64)
        # epdlogp = np.vstack(dlogps)
        epr =  np.asarray(np.vstack(drs), np.float64)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr, Config.GAMMA)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epx, epdlogp, model)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % Config.BATCH_SIZE == 0:
            for k, v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = Config.DECAY_RATE * rmsprop_cache[k] + (1 - Config.DECAY_RATE) * g**2
                model[k] += Config.LEARNING_RATE * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        
        if episode_number % GlobalConfig.CHECKPOINT_MARK == 0:
            FileManager.saveModel('PG', model)

        episode_number += 1

    progress_bar.finish()
    # print ('running mean: %f' % running_reward)
    print ('Finished in', time.time()-t, 'seconds ')
    
    FileManager.saveModel("PG", model)
    stock, trades = market.exportDataTradingData()
    FileManager.saveTrades(stock,trades)