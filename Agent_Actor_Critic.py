import os
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from Actor_Critic import ActorCriticNetwork

learning_rate=0.0003
gamma=0.99 # discount factor
n_actions=2
class Agent():
    def __init__(self,learning_rate,gamma,n_actions,):
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.n_actions=None
        self.action_space=[i for i in range(self.n_actions)]
        
        self.actor_critic=ActorCriticNetwork(num_action=n_actions)
        self.actor_critic=optim.Adam(self.actor_critic.parameters(),learning_rate)
        
    def choose_action(self,observation):
        state=torch.tensor([observation])
        _,probabilites=self.actor_critic(state)
        probabilites=torch.distributions.Categorical(probabilites)
        action=probabilites.sample()
        return action
        