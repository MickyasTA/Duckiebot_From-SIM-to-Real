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
        action=self.fc2(layer1)
        return action
    
class Agent():
    def __init__(self,input_dims,n_actions,lr,gamma=0.99,
                 epsilon=1.0,eps_dec=1e-5,eps_min=0.01):
        self.input_dims=input_dims
        self.n_actions=n_actions
        self.lr=lr
        self.epsilon=epsilon
        self.gamma=gamma
        self.eps_dec=eps_dec
        self.eps_min=eps_min
        
        #This makes action selection easear in our choose action function
        self.action_space=[i for i in range(self.n_actions)]
        # Define the Q value of our Agent 
        self.Q = LinearNetworkModel(self.lr,self.n_actions,self.input_dims)
        
        
    def choose_action(self,observation):
        
        if np.random.random()>self.epsilon:
            state=torch.tensor(observation,dtype=torch.float).to(self.Q.device)
            actions=self.Q.forward(state)
            action=torch.argmax(actions).item()# .item() is to get the value of an actual action to get the numpy array w/c is compatable with the gym environment 
        else:
            action=np.random.choice(self.action_space)
        return action
    
    def decrement_epsilon(self):
            self.epsilon= self.epsilon-self.eps_dec \
                if self.epsilon>self.eps_min else self.eps_min
    def learn(self,state,action,reward,state_new):
        self.Q.optimizer.zero_grad()
        states=torch.tensor(state,dtype=torch.float.to(self.Q.device))
        actions=torch.tensor(action).to(self.Q.device)
        rewards=torch.tensor(reward).to(self.Q.device)
        state_new=torch.tensor(state_new,dtype=torch.float).to(self.Q.device)
        
        q_pred=self.Q.forward(states)[actions]
        q_next =self.Q.forward(state_new).max()
        
        q_target=reward+self.gamma*q_next # the targe t we want to move is R+r* max(Q(s',a'))
        
        # define the loss
        loss=self.Q.Loss(q_target,q_pred).to(self.Q.device)
        
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()