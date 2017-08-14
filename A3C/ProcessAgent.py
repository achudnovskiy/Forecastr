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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time
from Common.Config import Config as GlobalConfig
from .Config import Config

from .Experience import Experience
from Common.Market import Market

class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.market = Market()
        self.actions = np.arange(Config.ACTIONS)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

        self.current_state = None
        self.previous_state = None

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            # r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            r = experiences[t].reward
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(Config.ACTIONS)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if GlobalConfig.FORECAST_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        self.market.reset()
        done = False
        experiences = []

        # time_count = 0
        reward_sum = 0.0

        positions = []

        while not done:
            # very first few frames
            if self.current_state is None:
                observation, reward, done, info = self.market.takeAction(0)  # 0 == NOOP
                self.current_state = observation.reshape(GlobalConfig.SLICE_SIZE*GlobalConfig.SLICE_WIDTH, 1, 1)
                continue

            prediction, value = self.predict(self.current_state)
            action = self.select_action(prediction)
            self.previous_state = self.current_state
            observation, reward, done, info = self.market.takeAction(action)
            self.current_state = observation.reshape(GlobalConfig.SLICE_SIZE*GlobalConfig.SLICE_WIDTH, 1, 1)
            positions.append(info)

            reward_sum += reward
            exp = Experience(self.previous_state, action, prediction, reward, done)
            experiences.append(exp)

            # if done or time_count == Config.TIME_MAX:
            if done:
                terminal_reward = 0 if done else value
                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                # time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0
                
            # time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            # total_reward = 0
            # total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                # total_reward += reward_sum
                # total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), self.market.totalReward))
