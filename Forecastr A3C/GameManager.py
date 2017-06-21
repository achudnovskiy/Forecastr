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

# import gym

from pandas_datareader import data as dreader
from sklearn import preprocessing
import numpy as np

from Config import Config

class GameManager:

    # @staticmethod
    # def load_data():
    #     return dreader.DataReader('INTC', 'google', Config.STOCK_START_DATE, Config.STOCK_END_DATE)

    game_data = dreader.DataReader('INTC', 'google', Config.STOCK_START_DATE, Config.STOCK_END_DATE)


    def __init__(self):
        # self.game_name = game_name
        # self.display = display
        self.iterator = Config.SLICE_SIZE
        self.current_position = 0
        # self.env = gym.make(game_name)

        self.data = GameManager.game_data
        self.dataLength = self.data.shape[0]
        self.dataWidth = self.data.shape[1]
        self.positions = []
        self.reset()

    def reset(self):

        # observation = self.env.reset()
        self.iterator = Config.SLICE_SIZE

    def get_num_actions(self):
        return Config.ACTIONS
    
    def prepro(self, iterator):
        #get stocks data from
        start_index = iterator - Config.SLICE_SIZE
        end_index = iterator
        data_to_process = self.data.values[start_index:end_index]
        scaled_data = preprocessing.MinMaxScaler().fit_transform(data_to_process)
        flatten_data = scaled_data.flatten().T
        # flatten_data = data_to_process.flatten().T
        return flatten_data

    def step(self, action):
        # self._update_display()
        stockData = self.prepro(self.iterator)
        cur_price = self.data['Close'][self.iterator]
        decision = 0
        reward = 0
        if self.current_position == 0:
            if action == 1:
                self.positions.append({'ts':self.iterator, 'ps':'open'})
                self.current_position = cur_price
                decision = 1
        elif action == 2:
            self.positions.append({'ts':self.iterator, 'ps':'close'})
            reward = cur_price - self.current_position
            self.current_position = 0
        
        data_with_position = np.append(stockData, np.float32(decision))
        observation = np.reshape(data_with_position, (-1, 1, 1))

        done = self.iterator == self.dataLength - 1
        info = {}
        self.iterator += 1

        return observation, reward, done, info

    # def _update_display(self):
    #     if self.display:
    #         self.env.render()


# GameManager().step(0)