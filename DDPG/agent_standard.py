import numpy as np
import torch
import collections
import copy
import torch.nn.functional as F


class DDPG_Agent():

    def __init__(self,
                 action_scale: torch.tensor,
                 action_upper_bound: np.ndarray,
                 action_lower_bound: np.ndarray,
                 replay_buffer,
                 replay_start_size: int,
                 batch_size: int,
                 actor_update_frequent: int,  # The frequency of updating the actor network
                 target_update_tau: float,  # The parameter for soft update
                 actor_network: torch.nn,
                 critic_network: torch.nn,
                 actor_optimizer: torch.optim,
                 critic_optimizer: torch.optim,
                 gamma: float = 0.9,
                 sigma_noise: float = 0.2,
                 device: torch.device = torch.device("cpu")
                 ) -> None:
        self.device = device

        self.exp_counter = 0

        self.replay_buffer = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.actor_update_frequent = actor_update_frequent

        self.target_update_tau = target_update_tau

        self.main_critic_network = critic_network.to(self.device)
        self.target_critic_network = copy.deepcopy(critic_network).to(self.device)
        self.main_actor_network = actor_network.to(self.device)
        self.target_actor_network = copy.deepcopy(actor_network).to(self.device)

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

        self.gamma = gamma
        self.sigma_noise = sigma_noise
        self.action_scale = action_scale
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound

    def get_behavior_action(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.main_actor_network(obs)
        action += torch.normal(0, self.action_scale*self.sigma_noise)
        action = action.cpu().detach().numpy().clip(self.action_lower_bound, self.action_upper_bound)

        return action

    def soft_update_network(self, main_network: torch.nn, target_network: torch.nn) -> None:
        """Soft update the parameters of the target network"""
        for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(self.target_update_tau*main_param.data+(1.0-self.target_update_tau)*target_param.data)

    def batch_Q_approximation(self,
                              obs: torch.tensor,
                              action: torch.tensor,
                              reward: torch.tensor,
                              next_obs: torch.tensor,
                              done: torch.tensor) -> None:
        """To update the main critic network"""
        current_Q = self.main_critic_network(obs, action).squeeze(1)
        TD_target = reward+(1-done)*self.gamma*self.target_critic_network(next_obs,
                                                                          self.target_actor_network(next_obs)).squeeze(1)
        critic_loss = torch.mean(F.mse_loss(current_Q,
                                            TD_target.detach()))  # detach the TD_target for frozen the parameters of the target network

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def batch_actor_update(self, obs: torch.tensor) -> None:
        """To update the main actor network.
            However, before that, we need to freeze the parameters of the main critic network"""
        for p in self.main_critic_network.parameters():
            p.requires_grad = False

        actor_loss = torch.mean(-self.main_critic_network(obs, self.main_actor_network(obs)))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """After updating the main actor network, 
            we need to unfreeze the parameters of the main critic network"""
        for p in self.main_critic_network.parameters():
            p.requires_grad = True

    def Q_approximation(self,
                        obs: np.ndarray,
                        action: int,
                        reward: float,
                        next_obs: np.ndarray,
                        done: bool) -> None:
        """Here, we continue use the framework of DQN, since DDPG is an extension of DQN.
            The max Q value is approximated by the target network."""

        self.exp_counter += 1
        self.replay_buffer.append((obs, action, reward, next_obs, done))

        if len(self.replay_buffer) > self.replay_start_size:
            obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
            self.batch_Q_approximation(obs, action, reward, next_obs, done)  # train critic
            if self.exp_counter%self.actor_update_frequent == 0:  # update actor network every actor_update_frequent steps
                self.batch_actor_update(obs)  # train actor
                self.soft_update_network(self.main_critic_network,
                                         self.target_critic_network)  # soft update target networks
                self.soft_update_network(self.main_actor_network, self.target_actor_network)
