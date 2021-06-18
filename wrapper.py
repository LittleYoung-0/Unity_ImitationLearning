# LittleYoung：对朕煜师兄的代码进行了学习和更改
# date:2021-04-19
# 【有一个数据维数在报错】

import logging
import itertools

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

# LY+
import argparse
import pickle
from collections import namedtuple
from itertools import count
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from tensorboardX import SummaryWriter

logger = logging.getLogger('UnityWrapper')
logger.setLevel(level=logging.INFO)

class UnityWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 no_graphics=True,
                 seed=None,
                 scene=None,
                 n_agents=1):
        self.scene = scene

        seed = seed if seed is not None else np.random.randint(0, 65536)

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self._env = UnityEnvironment(file_name=file_name,
                                     base_port=base_port,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed,
                                     additional_args=['--scene', scene, '--n_agents', str(n_agents)],
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel])
        if train_mode:
            self.engine_configuration_channel.set_configuration_parameters(width=200,
                                                                           height=200,
                                                                           quality_level=0,
                                                                           time_scale=100)
        else:
            self.engine_configuration_channel.set_configuration_parameters(width=1028,
                                                                           height=720,
                                                                           quality_level=5,
                                                                           time_scale=5,
                                                                           target_frame_rate=60)
        self._env.reset()
        self.behavior_name = list(self._env.behavior_specs)[0]

    def init(self):
        behavior_specs = self._env.behavior_specs[self.behavior_name]
        logger.info(f'Observation shapes: {behavior_specs.observation_specs}')

        self._empty_action = behavior_specs.action_spec.empty_action

        discrete_action_size = 0
        if behavior_specs.action_spec.discrete_size > 0:
            discrete_action_size = 1
            action_product_list = []
            for action, branch_size in enumerate(behavior_specs.action_spec.discrete_branches):
                discrete_action_size *= branch_size
                action_product_list.append(range(branch_size))
                logger.info(f"Discrete action branch {action} has {branch_size} different actions")

            self.action_product = np.array(list(itertools.product(*action_product_list)))

        continuous_action_size = behavior_specs.action_spec.continuous_size

        logger.info(f'Continuous action sizes: {continuous_action_size}')

        self.d_action_dim = discrete_action_size
        self.c_action_dim = continuous_action_size

        for o in behavior_specs.observation_specs:
            if len(o) >= 3:
                self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                break

        return behavior_specs.observation_specs, discrete_action_size, continuous_action_size

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)

        return [obs.astype(np.float32) for obs in decision_steps.obs]

    def step(self, d_action, c_action):
        if self.d_action_dim:
            d_action = np.argmax(d_action, axis=1)
            d_action = self.action_product[d_action]

        self._env.set_actions(self.behavior_name,
                              ActionTuple(continuous=c_action, discrete=d_action))

        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        tmp_terminal_steps = terminal_steps

        while len(decision_steps) == 0:
            self._env.set_actions(self.behavior_name, self._empty_action(0))
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
            tmp_terminal_steps.agent_id = np.concatenate([tmp_terminal_steps.agent_id,
                                                          terminal_steps.agent_id])
            tmp_terminal_steps.reward = np.concatenate([tmp_terminal_steps.reward,
                                                        terminal_steps.reward])
            tmp_terminal_steps.interrupted = np.concatenate([tmp_terminal_steps.interrupted,
                                                             terminal_steps.interrupted])

        reward = decision_steps.reward
        reward[tmp_terminal_steps.agent_id] = tmp_terminal_steps.reward

        done = np.full([len(decision_steps), ], False, dtype=np.bool)
        done[tmp_terminal_steps.agent_id] = True

        max_step = np.full([len(decision_steps), ], False, dtype=np.bool)
        max_step[tmp_terminal_steps.agent_id] = tmp_terminal_steps.interrupted

        return ([obs.astype(np.float32) for obs in decision_steps.obs],
                decision_steps.reward.astype(np.float32),
                done,
                max_step)

    def close(self):
        self._env.close()


# LY+Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(30, num_action)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# LY+DQN

class DQN():
    update_count = 0
    memory_count = 0
    def __init__(self):
        self.act_net, self.target_net = Net(), Net()
        self.memory = np.zeros((MEMORY_CAPACITY, num_state * 2 + num_action +1))
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.act_net.parameters(), lr= 0.01)
        self.loss = nn.MSELoss()

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, action, reward, next_state))
        self.memory[index] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= 0.9:
            action_value =self.act_net.forward(state)
            action = action_value.detach().numpy()
        else:
            action = np.random.randn(n_agents, c_action_dim)
        return action

    def learn(self):
        if self.learn_counter % 100 == 0:
            self.target_net.load_state_dict(self.act_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :num_state])
        batch_action = torch.LongTensor(batch_memory[:, num_state:num_state+2])
        batch_reward = torch.FloatTensor(batch_memory[:, num_state+2: num_state+3])
        batch_next_state = torch.FloatTensor(batch_memory[:, -num_state:])

        q_act = self.act_net(batch_state).gather(1, batch_action)   # gather的使用：https://zhuanlan.zhihu.com/p/110289027
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA*q_next.max(1)[0].view(batch_size, 1)

        loss = self.loss(q_act, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":

    MaxEpisode = 100
    MaxStep = 100
    logging.basicConfig(level=logging.DEBUG)
    MEMORY_CAPACITY = 2000
    batch_size = 32
    GAMMA = 0.9

    env = UnityWrapper(train_mode=True, base_port=5004)
    obs_shape_list, d_action_dim, c_action_dim = env.init()

    num_action = c_action_dim
    num_state = 8  # 这个是obs_shape_list的shape

    net =DQN()
    step_counter_list = []
    for episode in range(MaxEpisode):
        state = env.reset()
        step_counter = 0
        n_agents = state[0].shape[0]  # print("the number of agents is:", n_agents)
        while True:
            d_action, c_action = None, None
            step_counter += 1
            c_action = net.choose_action(state)
            c_action = np.random.randn(n_agents, c_action_dim)
            next_state, reward, done, max_step = env.step(d_action, c_action)
            # net.store_trans(state, c_action, reward, next_state)
            state = list(np.array(state).flatten())
            next_state = list(np.array(next_state).flatten())
            c_action = list(np.array(c_action).flatten())
            reward = list(np.array(reward).flatten())
            net.store_trans(state, c_action, reward, next_state)
            if net.memory_counter >= MEMORY_CAPACITY:
                net.learn()
                if done:
                    print("episode {}, the reward is {}".format(episode, round(reward, 3)))
            if done:
                step_counter_list.append(step_counter)
                # plot
                break
            state = next_state

    env.close()





