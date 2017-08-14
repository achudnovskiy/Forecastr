from Common.Config import Config as GlobalConfig


class Config:

    HIDDEN_LAYER_COUNT = 200 # number of hidden layer neurons
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 10 # every how many episodes to do a param update?
    GAMMA = 0.99 # discount factor for reward
    DECAY_RATE = 0.99 # decay factor for RMSProp leaky sum of grad^2