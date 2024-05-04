import torch 
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim 
import torch.nn.functional as F

class LinearNetworkModel(nn.Module):
    def __init__(self,lr,num_action,imput_dim):
        super(LinearNetworkModel).__init__()
        self.fc1=nn.Linear(imput_dim,128)
        self.fc2=nn.Linear(128,num_action)
        self.optimizer=torch.optim.Adam(nn.Parameter(),lr)
        self.Loss=nn.MSELoss()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)       
        
    def forward(self,state):
        layer1=F.relu(self.fc1(state))
        layer2=F.relu(self.fc2(layer2))
        return layer2