import math

import torch
import copy
import torch.nn.functional as F
import numpy as np
import collections


# 感知体Agent，属性包括state维度，action维度，学习率lr，account ratio gamma，探险系数e_greed
class DDPG_Agent:
    def __init__(self,
                 critic_network,
                 actor_network,
                 critic_optimizer,  # 优化器，即神经网络参数更新方法
                 actor_optimizer,
                 replay_buffer,  # 经验回放池
                 batch_size,  # 进行一次训练时从经验池提取的样本数量
                 replay_start_size,  # 开始利用经验池时经验池的样本数量
                 action_scale,
                 action_upper_bound,
                 action_lower_bound,
                 actor_update_frequent,
                 target_update_tau,  # 软更新权重系数
                 sigma_noise=0.2,  # 噪声系数
                 gamma=0.9  # 奖励折扣率
                 ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 神经网络相关参数
        self.main_critic_network = critic_network.to(self.device)  # 求pre_Q时使用，一次batch训练更新一次
        self.target_critic_network = copy.deepcopy(critic_network).to(self.device)  # 求target_Q时使用，多次训练后再更新
        self.main_actor_network = actor_network.to(self.device)
        self.target_actor_network = copy.deepcopy(actor_network).to(self.device)
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        # 经验池相关参数
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        # 环境相关参数
        self.action_scale = action_scale
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound

        self.sigma_noise = sigma_noise
        self.gamma = gamma
        self.exp_counter = 0  # episode计数
        self.actor_update_frequent = actor_update_frequent
        self.target_update_tau = target_update_tau

    # 根据经验选择action
    def get_behavior_action(self, obs, step):
        if step == 0:
            action = [0, math.pi/2]
        else:
            obs = torch.as_tensor(obs.reshape(1, 1, 24, 24, 24), dtype=torch.float32).to(self.device)
            action = self.main_actor_network(obs).cpu()  # 输入状态值，输出确定的action
            # print(torch.normal(0, self.action_scale*self.sigma_noise))
            action += torch.normal(0, self.action_scale * self.sigma_noise)  # 返回正态分布的随机张量;目的为给action添加噪声,增加探索性
            action = action.detach().cpu().numpy().clip(self.action_lower_bound, self.action_upper_bound)  # 限定action在范围内
        return action

    # 神经网络参数软更新
    def soft_update_network(self, main_network, target_network):
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(
                self.target_update_tau * main_param.data + (1 - self.target_update_tau) * target_param.data)

    # update main critic network
    def batch_Q_approximation(self, obs, action, reward, next_obs, done, batch_size):
        current_Q = self.main_critic_network(obs.reshape(batch_size,1,24, 24, 24), action).squeeze(1)  # squeeze将shape为1的维度去掉
        next_station = self.target_actor_network(next_obs.reshape(batch_size,1,24,24,24)).to(self.device)
        TD_target = reward + (1 - done) * self.gamma * self.target_critic_network(next_obs.reshape(batch_size,1,24, 24, 24), next_station).squeeze(1)

        critic_loss = torch.mean(F.mse_loss(current_Q, TD_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # update main actor network
    def batch_actor_update(self, obs, batch_size):
        for p in self.main_critic_network.parameters():
            p.requires_grad = False
        action = self.main_actor_network(obs.reshape(batch_size,1,24,24,24))
        # print(self.main_critic_network(obs, action))
        actor_loss = torch.mean(-self.main_critic_network(obs, action))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.main_critic_network.parameters():
            p.requires_grad = True

    # 对整体进行学习
    def Q_approximation(self, obs, action, reward, next_obs, done, batch_size):
        self.exp_counter += 1
        self.replay_buffer.append((obs.reshape(1,24,24,24), action, reward, next_obs.reshape(1,24,24,24), done))  # 将样本加入经验池
        # 当经验池样本数量高于开始训练要求数量，每放入num_steps个样本进行一次批量训练
        if len(self.replay_buffer) > self.replay_start_size:
            obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)  # 在经验池中随机取样batch_size个进行训练
            # print('obs'+str(obs))
            # print('action'+str(action))
            self.batch_Q_approximation(obs, action, reward, next_obs, done, batch_size)  # 对main_critic_network进行训练
            if self.exp_counter % self.actor_update_frequent == 0:  # episode达到actor_update_frequent时,对target_network进行参数更新
                self.batch_actor_update(obs, batch_size)  # 对main_actor_network进行训练
                self.soft_update_network(self.main_critic_network,
                                         self.target_critic_network)  # soft update target networks
                self.soft_update_network(self.main_actor_network, self.target_actor_network)
