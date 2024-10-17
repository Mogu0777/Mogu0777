import gymnasium as gym
import agent
import torch
import mould
import reply_buffers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import NBV_Env
import random
import math

"""
    参考：https://www.jianshu.com/p/22cdc0d9fa13;
    https://zhuanlan.zhihu.com/p/212187310;
    https://zhuanlan.zhihu.com/p/111257402;
"""
class TrainManager:
    def __init__(self,
                 env,  # 环境
                 episode_num=100000,  # 轮次数量
                 critic_lr=0.001,  # critic学习率 0.0001
                 actor_lr=0.0001,  # actor学习率 0.00001
                 gamma=1,  # 奖励折扣率0.99
                 sigma_noise=0.75,  # 噪声系数 0.8
                 memory_size=5000,  # 经验池容量 8000
                 replay_start_size=200,  # 开始训练时所需经验池样本数量
                 batch_size=64,  # 进行一次训练时从经验池提取的样本数量
                 actor_update_frequent=1,
                 target_update_tau=0.005,  # 软更新权重系数 0.005
                 noise_dec=0.000008,  # 原0.0005
                 noise_min=0.01,  # 原0.5
                 my_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 seed=0
                 ):

        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(self.seed)
        self.device = torch.device(my_device)
        self.batch_size = batch_size

        self.env = env
        self.episode_num = episode_num

        self.noise_dec = noise_dec
        self.noise_min = noise_min

        obs_dim = self.env.state.shape[0]
        action_dim = gym.spaces.utils.flatdim(self.env.action_space)

        action_upper_bound = env.action_space.high
        action_lower_bound = env.action_space.low
        action_bias = (action_upper_bound + action_lower_bound) / 2.0
        action_bias = torch.tensor(action_bias, dtype=torch.float32)
        action_scale = (action_upper_bound - action_lower_bound) / 2.0
        action_scale = torch.tensor(action_scale, dtype=torch.float32)

        critic_network = mould.critic_network(obs_dim, action_dim)
        actor_network = mould.actor_network(obs_dim, action_dim, action_scale.to(self.device),
                                            action_bias.to(self.device))
        critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=critic_lr)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=actor_lr)

        replay_buffer = reply_buffers.ReplayBuffer(max_size=memory_size)  # 经验池
        self.agent = agent.DDPG_Agent(
            critic_network=critic_network,
            actor_network=actor_network,
            # critic_network=torch.load('target_critic_network.pt'),
            # actor_network=torch.load('target_actor_network.pt'),
            critic_optimizer=critic_optimizer,  # 优化器，即神经网络参数更新方法
            actor_optimizer=actor_optimizer,
            replay_buffer=replay_buffer,  # 经验回放池
            batch_size=batch_size,  # 进行一次训练时从经验池提取的样本数量
            replay_start_size=replay_start_size,  # 开始利用经验池时经验池的样本数量
            action_scale=action_scale,
            action_upper_bound=action_upper_bound,
            action_lower_bound=action_lower_bound,
            actor_update_frequent=actor_update_frequent,
            target_update_tau=target_update_tau,  # 软更新权重系数
            sigma_noise=sigma_noise,  # 噪声系数
            gamma=gamma  # 奖励折扣率
        )
        self.episode_total_rewards = np.zeros(self.episode_num)
        self.index_episode = 0

    def save_mould(self):
        torch.save(self.agent.target_critic_network, 'target_critic_network.pt')
        torch.save(self.agent.target_actor_network, 'target_actor_network.pt')

    def load_mould(self):
        new_m = torch.load('rnn1.pt')
        return new_m

    # 一个episode的训练
    def train_episode(self):
        total_reward = 0
        obs, epi_step = self.env.reset()
        while True:
            obs = torch.as_tensor(obs, dtype=torch.float).detach()
            action = self.agent.get_behavior_action(obs.to(self.device), step=epi_step)  # 根据当前状态获取action
            action = np.squeeze(action)
            next_obs, reward, done, epi_step = self.env.step(action)  # 根据action获取下一状态
            self.agent.Q_approximation(obs, action, reward, next_obs, done, self.batch_size)  # 进行参数更新
            obs = next_obs
            total_reward += reward
            # _,_ = self.env.render()
            if done:
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                self.agent.sigma_noise = self.agent.sigma_noise - self.noise_dec if self.agent.sigma_noise > self.noise_min else self.noise_min
                break
        return total_reward

    # 测试episode显示
    def test_episode(self, save_result = False):
        total_reward = 0
        obs, epi_step = self.env.reset()
        result = []
        while True:
            obs = torch.as_tensor(obs.reshape(1, 1, 24,24,24), dtype=torch.float).to(self.device)
            if epi_step==0:
                action = [0, math.pi/2]
            else:
                action = self.agent.target_actor_network(obs).detach()
                action = np.squeeze(action)
            next_obs, reward, done, epi_step = self.env.step(action)
            obs = next_obs
            total_reward += reward
            a_angle, e_angle = self.env.render()
            result.append([a_angle, e_angle])
            if done:
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                break
        if save_result:
            print(result)
            # np.savetxt("results.txt", np.array(result), fmt='%f', delimiter=',')
        return total_reward

    # 多次episode训练
    def train(self):
        for e in range(self.episode_num):
            if e==self.episode_num-1:
                episode_reward = self.test_episode(save_result=True)
                print('Test Episode %s: reward= %.2f' % (e, episode_reward))
                print('e_greed= %f' % self.agent.sigma_noise)
            elif (e + 1) % 50 == 0:
                episode_reward = self.test_episode()
                tm.save_mould()
                print('Test Episode %s: reward= %.2f' % (e, episode_reward))
                print('e_greed= %f' % self.agent.sigma_noise)
            else:
                episode_reward = self.train_episode()
                print('Episode %s: reward= %.2f' % (e, episode_reward))
            # if e==500:
            #     self.agent.sigma_noise = 0.5

    def ceshi(self):
        total_reward = 0
        obs, epi_step = self.env.reset()
        result = []
        while True:
            obs = torch.as_tensor(obs.reshape(1, 1, 24, 24, 24), dtype=torch.float).to(self.device)
            if epi_step == 0:
                action = [0, math.pi / 2]
            else:
                action = self.agent.target_actor_network(obs).detach()
                action = np.squeeze(action)
            next_obs, reward, done, epi_step = self.env.step(action)
            obs = next_obs
            total_reward += reward
            a_angle, e_angle, cover = self.env.render()
            result.append(cover)
            if done:
                self.episode_total_rewards[self.index_episode] = total_reward
                self.index_episode += 1
                break
        print(result)
        print(np.trapz(result))
        plt.plot(result,'bo-', label="Episode Reward")
        plt.show()
        return total_reward

    def plotting(self, smoothing_window: int = 100) -> None:
        """ Plot the episode reward over time. """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.episode_total_rewards, label="Episode Reward")
        # Use rolling mean to smooth the curve
        rewards_smoothed = pd.Series(self.episode_total_rewards).rolling(smoothing_window,
                                                                         min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed, label="Episode Reward (Smoothed)")
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title("Episode Reward over Time")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    env1 = NBV_Env.nbv_env()
    tm = TrainManager(env1)
    # tm.ceshi()
    tm.train()
    tm.save_mould()
    tm.plotting()
#