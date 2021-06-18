# data:2021-06-16
# LittleYoung
# 参考：方块环境下的GAIL代码(离散动作，LittleYoungのgitub)，基于gym的GAIL代码(连续动作，【github地址】)，使用DDPG与Unity进行交互训练的代码(LittleYoungの私人gitub项目)
# 使用GAIL算法与Ml-agents环境进行交互
# 分为三步①使用DDPG算法训练，并记录，作为专家示例，②使用GAIL训练
# 本代码文件作为②的主函数，调用GAIL算法进行训练


import torch
import numpy as np
from GAIL_algo import GAIL
import logging
from wrapper import UnityWrapper

logging.basicConfig(level=logging.DEBUG)

def train():
    MaxEpoch = 1000
    MaxEpisode = 1000
    MaxStep = 1000
    n_iter = 100
    batch_size = 100

    env = UnityWrapper(train_mode=True, base_port=5004)
    obs_shape_list, action_d_dim, action_c_dim = env.init()
    StateDim = 8
    ActionDim = action_c_dim
    max_action = 1

    expert_state_file = 'PPO_Unity0618_state.npy'
    expert_action_file = 'PPO_Unity0618_action.npy'

    policy = GAIL(StateDim=StateDim, ActionDim=ActionDim, max_action=max_action, expert_state_file=expert_state_file, expert_action_file=expert_action_file)

    epochs = []
    rewards = []

    for epoch in range(MaxEpoch):
        policy.update(n_iter, batch_size)

        for episode in range(MaxEpisode):
            state = env.reset()
            episode_reward = 0
            for t in range(MaxStep):
                action = policy.choose_action(state)
                next_state, reward, done, max_step = env.step(None, action)
                episode_reward += reward
                if done:
                    break
            print('The epoch is:', epoch, ' The episode is:', episode, ' The episode reward is:',episode_reward)

    policy.save()

if __name__ == '__main__':
    train()




