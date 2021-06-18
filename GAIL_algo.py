# data:2021-06-16
# LittleYoung
# 使用GAIL算法与Ml-agents环境进行交互
# 分为三步①使用DDPG算法训练，并记录，作为专家示例，②使用GAIL训练
# 本代码文件作为②的子模块，单独封装了GAIL算法

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import sample

# 生成器，前向传播，产生某个状态下的动作
class Actor(nn.Module):
    def __init__(self, StateDim, ActionDim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(StateDim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, ActionDim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

# 判别器，判别生成器轨迹序列与专家轨迹序列的相似度
class Discriminator(nn.Module):
    def __init__(self, StateDim, ActionDim):
        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(StateDim + ActionDim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class GAIL(object):
    def __init__(self, StateDim, ActionDim, max_action, expert_state_file, expert_action_file):
        self.actor = Actor(StateDim, ActionDim, max_action)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)   # 这里的一些参数还不确定【】

        self.discriminator = Discriminator(StateDim, ActionDim)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        self.expert_state = np.load(expert_state_file)
        self.expert_action = np.load(expert_action_file)

        self.loss_fn = nn.BCELoss()
        self.max_action = max_action

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        action = action.detach().numpy()
        action = np.squeeze(action, axis=0)
        return action

    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            # 从专家数据集中进行采样
            sample_expert_state = self.expert_state.tolist()
            sample_expert_state = sample(sample_expert_state, batch_size)
            sample_expert_action = self.expert_action.tolist()
            sample_expert_action = sample(sample_expert_action, batch_size)

            # Actor生成算法的数据
            sample_state = torch.FloatTensor(sample_expert_state)   # 因为要传入网络，所以先做一个数据转换
            sample_action = self.actor(sample_state)

            ########################
            # 更新判别器参数
            ########################
            self.optim_discriminator.zero_grad()

            # 全对的情况
            expert_label = torch.full((batch_size, 1), 1.0)
            policy_label = torch.full((batch_size, 1), 0.0)

            # 计算loss
            sample_expert_state = torch.FloatTensor(sample_expert_state)
            sample_expert_action = torch.FloatTensor(sample_expert_action)

            prob_expert = self.discriminator(sample_expert_state, sample_expert_action)
            loss = self.loss_fn(prob_expert, expert_label)

            prob_policy = self.discriminator(sample_state, sample_action)
            loss = loss + self.loss_fn(prob_policy, policy_label)

            # 更新
            loss.backward(retain_graph= True)
            self.optim_discriminator.step()

            ########################
            # 更新生成器参数
            ########################
            self.optim_actor.zero_grad()

            loss_actor = -self.discriminator(sample_state, sample_action)
            loss_actor.mean().backward(retain_graph= True)
            self.optim_actor.step()

    def save(self, directory='trained', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, name))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory, name))

    def load(self, directory= './trained', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, name)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory, name)))






