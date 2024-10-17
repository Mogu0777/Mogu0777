import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class actor_network(nn.Module):
    def __init__(self, obs_dim, act_dim, action_scale, action_bias):
        super(actor_network, self).__init__()

        self.conv1 = nn.Conv3d(1, 10, (2, 2, 2), (1,1,1))
        self.conv2 = nn.Conv3d(10, 12, (2, 2, 2), (1,1,1))
        self.conv3 = nn.Conv3d(12, 8, (2, 2, 2), (1,1,1))

        self.fc1 = nn.Linear(1500,1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, act_dim)
        self.action_scale = action_scale  # (high-low)/2
        self.action_bias = action_bias  #(high+low)/2

    def forward(self, x):
        input_size = x.size(0)

        # print(x.cpu().numpy())
        x = self.conv1(x)

        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        x = x.view(input_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # print(x)
        x = torch.tanh(x)
        return x*self.action_scale+self.action_bias

class critic_network(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(critic_network, self).__init__()

        self.conv1 = nn.Conv3d(1, 10, (2, 2, 2), (1, 1, 1))
        self.conv2 = nn.Conv3d(10, 12, (2, 2, 2), (1, 1, 1))
        self.conv3 = nn.Conv3d(12, 8, (2, 2, 2), (1, 1, 1))

        self.fc1 = nn.Linear(1500 + act_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, obs, act):
        input_size = obs.size(0)
        # in: batch*1*28*28, out: batch*10*24*24(28-5+1)
        # print(x.cpu().numpy())
        x = self.conv1(obs)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        x = x.view(input_size, -1)

        x = torch.cat([x, act], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x