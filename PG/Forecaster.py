from pandas_datareader import data as dreader
import numpy as np
from sklearn import preprocessing
import progressbar
import time


def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def policy_forward(x,model):
    if(len(x.shape)==1):
        x = x[np.newaxis,...]
    
    h = x.dot(model['W1'])
    h[h<0] = 0 # ReLU nonlinearity
    logp = h.dot(model['W2'])
    p = softmax(logp)
    
    return p, h # return probability of taking action 2, and hidden state

start_period = '2010-01-01'
end_period = '2017-03-01'
model_file_name = 'top_secret_model'

dt,row_count,col_count = pulldata(start_period,end_period)

period_length = 30 # number of days of data in the buffer
HIDDEN_LAYER_COUNT = 200 # number of hidden layer neurons
DIMENSION = period_length * col_count + 1
learning_rate = 1e-4
batch_size = 10 # every how many episodes to do a param update?
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
ACTIONS = 3   # actions: 0 - do nothing, 1 - buy, 2 - sell


f = open(model_file_name, "rb")
model = pickle.load(f)

running_reward = None
episode_number = 0
positions = []

progress_bar = progressbar.ProgressBar(maxval=row_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
progress_bar.start()

    
cur_position = 0
reward_sum = np.float64(0)
i = period_length + 1

while i < row_count:
    reward = np.float64(0)
    x = prepro(i,dt)
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

    # step the environment and get new measurements
    if cur_position == 0:
        if action == 1:
            positions.append({'ts':i,'ps':'open'})
            cur_position = cur_price
    elif action == 2:
        positions.append({'ts':i,'ps':'close'})
        reward = cur_price - cur_position
        cur_position = 0
    
    reward_sum += reward
    i = i + 1
    progress_bar.update(i)

progress_bar.finish()
print ' reward: %f' % reward_sum


pl_dt = dt['Close']
ax = dt['Close'].plot(figsize=(20,10))
for pos in positions:
    act_ind = pos['ts']
    ax.annotate(pos['ps'],(dt.index[act_ind], dt['Close'][act_ind]), textcoords='offset points',xytext=(15, 15),arrowprops=dict(arrowstyle='-|>'))
df2.plot(ax=ax)