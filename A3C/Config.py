# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from Common.Config import Config as GlobalConfig

class Config:

  
    #########################################################################
    # Number of agents, predictors, trainers and other system settings
    
    # If the dynamic configuration is on, these are the initial values.
    # Number of Agents
    AGENTS = 1
    # Number of Predictors
    PREDICTORS = 1
    # Number of Trainers
    TRAINERS = 1

    # Device
    if GlobalConfig.USE_CUDA:
        Device = 'gpu:0'
    else:
        DEVICE = 'cpu:0'

    # Enable the dynamic adjustment (+ waiting time to start it)
    DYNAMIC_SETTINGS = True
    DYNAMIC_SETTINGS_STEP_WAIT = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNT = 0.99
    
    # Max size of the queue
    MAX_QUEUE_SIZE = 100
    PREDICTION_BATCH_SIZE = 128

    ACTIONS = GlobalConfig.ACTIONS

    # Total number of episodes and annealing frequency
    EPISODES = 400000
    ANNEALING_EPISODE_COUNT = 400000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.01
    BETA_END = 0.01

    # Learning rate
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.0003

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    # Dual RMSProp - we found that using a single RMSProp for the two cost function works better and faster
    DUAL_RMSPROP = False
    
    # Gradient clipping
    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0 
    # Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower
    TRAINING_MIN_BATCH_SIZE = 0
    
    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = True
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    
    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 100000
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 1000

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here
    
    # Minimum policy
    MIN_POLICY = 0.0
    # Use log_softmax() instead of log(softmax())
    USE_LOG_SOFTMAX = False
