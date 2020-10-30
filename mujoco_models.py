import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import gym  # open ai gym
import argparse


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim + action_dim, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.fc1 = nn.Linear(state_dim, 800)
        self.fc2 = nn.Linear(800, 400)

        self.fc_mean = nn.Linear(400, action_dim)
        self.fc_log_std = nn.Linear(400, action_dim)

        self.action_scale = 1.

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = self.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1-y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = self.tanh(mean) * self.action_scale

        return action, log_prob, mean

    def choose_action(self, obs):
        x = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, mean = self.forward(x)

        return action.squeeze().cpu().numpy(), log_prob, mean.squeeze().cpu().numpy()



