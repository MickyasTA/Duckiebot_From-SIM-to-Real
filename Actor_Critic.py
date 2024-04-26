import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCriticNetwork(nn.Module):
# since every thing is given except for the num_action we pass only num_action 
    def __init__(self,num_action,fc1_dims=1024,fc2_dims=512,
                 name="actor_critic",check_path_dir="tmp/actor_critic",):
        super(ActorCriticNetwork,self).__init__()
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.num_action=num_action
        self.model_name=name # we have to use model name b/c name is reserved by the base class
        self.checkpoint_dir=check_path_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,name+"_ac")
        
        # The network
        self.fc1=nn.Linear(in_features=self.num_action,out_features=self.fc1_dims)
        self.fc2=nn.Linear(in_features=self.fc1_dims,out_features=self.fc2_dims)
        self.value=nn.Linear(in_features=self.fc2_dims,out_features=1) # there is no activation 
        self.policy_pi=nn.Linear(self.fc2_dims,num_action)
        self.policy_act=nn.Softmax()
        
    def forward(self,state):
        x=self.fc1(state)
        x=nn.ReLU(x)
        x=self.fc2(x)
        x=nn.ReLU(x)
        
        value=self.value(x)
        policy_pi=self.policy_act(self.policy_pi(x))
        
        return value ,policy_pi
"""    check point of this code and it is working well 
model=ActorCriticNetwork(12)
print(model)
        """
    