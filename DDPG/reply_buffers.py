import random
import collections
import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = collections.deque(maxlen=max_size)  #经验池容量

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        # for i in obs_batch:
        #     print(type(i))
        obs_batch = torch.stack(obs_batch)
        # obs_batch = torch.FloatTensor(obs_batch)
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)
