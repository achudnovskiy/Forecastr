#!/usr/bin/python

import sys
from pandas_datareader import data as dreader
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
import numpy as np
# import skcuda.linalg as linalg
# import skcuda.misc as misc
from sklearn import preprocessing
import cPickle as pickle
import progressbar
import time


def pulldata(start_date, end_date):
    stock_data = dreader.DataReader('INTC', 'yahoo', start_date, end_date)
    return stock_data, stock_data.shape[0], stock_data.shape[1]

def initModel(D, H, A):
    model = {}
    model['W1'] = np.asarray(np.random.randn(D, H) / np.sqrt(D), np.float64) # "Xavier" initialization
    model['W2'] = np.asarray(np.random.randn(H, A) / np.sqrt(H), np.float64)

    return model

def restore_or_create_model(file_name, DIMENSION, HIDDEN_LAYER_COUNT, ACTIONS):
    try:
        with open(file_name, "rb") as f:
            m = pickle.load(f)
            print('restoring the model')
    except StandardError:
        m = initModel(DIMENSION, HIDDEN_LAYER_COUNT, ACTIONS)
        print('generating new model')
    return m

def prepro(i,all_data, period_length):
    #get stocks data from 
    start_index = i - period_length
    end_index =  i
    data_to_process = all_data.values[start_index:end_index]
    scaled_data = preprocessing.MinMaxScaler().fit_transform(data_to_process)
    flatten_data = scaled_data.flatten().T
    return flatten_data

def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def policy_forward(x, model):
    if len(x.shape) == 1:
        x = x[np.newaxis, ...]

    h = x.dot(model['W1'])
    # h = cuda_multiply(x.copy(), model['W1'])
    h[h<0] = 0 # ReLU nonlinearity
    logp = h.dot(model['W2'])
    # logp = cuda_multiply(h, model['W2'])
    p = softmax(logp)

    return p, h # return probability of taking action 2, and hidden state

def cuda_multiply(mtx1, mtx2, trans1='N', trans2='N'):
    mtx1_gpu = gpuarray.to_gpu(mtx1)
    mtx2_gpu = gpuarray.to_gpu(mtx2)
    return linalg.dot(mtx1_gpu, mtx2_gpu, transa=trans1, transb=trans2).get()

def policy_backward(eph, epx, epdlogp, model):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = eph.T.dot(epdlogp)
    # dW2 = cuda_multiply(eph, epdlogp, 'T')
    #dW2 = np.dot(eph.T, epdlogp).ravel()

    dh = epdlogp.dot(model['W2'].T)
    # dh = cuda_multiply(epdlogp, model['W2'], 'N', 'T')
    # dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu

    
    dW1 = epx.T.dot(dh)
    # dW1 = cuda_multiply(epx, dh, 'T')

    return {'W1':dW1, 'W2':dW2}

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def restore_model(file_name):
    fileObject = open(file_name, 'r')
    return pickle.load(fileObject)

def save_model(file_name, object_to_save):
    fileObject = open(file_name, 'wb')
    pickle.dump(object_to_save, fileObject)
    fileObject.close()


def main():
    # linalg.init()

    start_period = '2010-01-01'
    end_period = '2017-03-01'
    model_file_name = 'top_secret_model'
    episode_count = 50000

    dt, row_count, col_count = pulldata(start_period, end_period)

    period_length = 30 # number of days of data in the buffer
    HIDDEN_LAYER_COUNT = 200 # number of hidden layer neurons
    DIMENSION = period_length * col_count + 1
    learning_rate = 1e-4
    batch_size = 10 # every how many episodes to do a param update?
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    ACTIONS = 3   # actions: 0 - do nothing, 1 - buy, 2 - sell

    model = restore_or_create_model(model_file_name, DIMENSION, HIDDEN_LAYER_COUNT, ACTIONS)
    grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k, v in model.iteritems() } # rmsprop memory

    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    t = time.time()
    te = time.time()

    progress_bar = progressbar.ProgressBar(maxval=episode_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    progress_bar.start()

    while episode_number < episode_count:

        print 'processing',episode_number,'elapsed', time.time()-te, 'seconds '
        te = time.time()
        
        i = period_length + 1
        cur_position = 0
        reward_sum = np.float64(0)

        while i < row_count:
            reward = np.float64(0)
            x = prepro(i,dt, period_length)
            cur_price = dt['Adj Close'][i]
            if cur_position != 0:
                x = np.append(x,np.float64(1))
            else:
                x = np.append(x,np.float64(0))
            # forward the policy network and sample an action from the returned probability
            aprob, h = policy_forward(x,model)
            # roll the dice, in the softmax loss
            u = np.random.uniform()
            aprob_cum = np.cumsum(aprob)
            action = np.where(u <= aprob_cum)[0][0]

            # record various intermediates (needed later for backprop)
            xs.append(x) # observation
            hs.append(h) # hidden state

            #softmax loss gradient
            dlogsoftmax = aprob.copy()
            dlogsoftmax[0,action] -= 1 #-discouted reward 
            dlogps.append(dlogsoftmax)

            # step the environment and get new measurements
            if cur_position == 0:
                if action == 1:
                    cur_position = cur_price
            elif action == 2:
                reward = cur_price - cur_position
                cur_position = 0

            reward_sum += reward
            drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
            i = i + 1


        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.asarray(np.vstack(xs), np.float64)
        eph = np.asarray(np.vstack(hs), np.float64)

        epdlogp = np.asarray(np.vstack(dlogps), np.float64)
        # epdlogp = np.vstack(dlogps)
        epr =  np.asarray(np.vstack(drs), np.float64)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr, gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epx, epdlogp, model)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        episode_number += 1
        progress_bar.update(episode_number)

    progress_bar.finish()
    print ' running mean: %f' % running_reward
    print 'Finished in', time.time()-t, 'seconds '

    save_model(model_file_name, model)



if __name__ == "__main__":
    main()
