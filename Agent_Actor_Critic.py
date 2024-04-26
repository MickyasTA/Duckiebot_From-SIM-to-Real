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
    def __init__(self,learning_rate=0.0003,gamma=0.99,n_actions=2,):
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.actions=None
        self.n_actions=n_actions
        self.action_space=[i for i in range(self.n_actions)] # self.n_actions.n b/c  gym environment  want to get the number of actions
        
        self.actor_critic=ActorCriticNetwork(num_action=n_actions)
        self.optimizer=optim.Adam(self.actor_critic.parameters(),learning_rate)
        
    def choose_action(self,observation):
        state=torch.tensor(observation, dtype=torch.float32).unsqueeze(0) 
        _,probabilites=self.actor_critic.forward(state)
        probabilites=torch.distributions.Categorical(probabilites)
        action=probabilites.sample()
        self.action=action
        return action.numpy()[0] # we return the action as a numpy array
    def save_model(self):
        print("... saving model ...")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
    def load_model(self):
        print("... loading model ...")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
    def learn(self,state,reward,state_,done,):
        state=torch.tensor(state)
        reward=torch.tensor(reward,dtype=torch.float32) # dtype=torch.float32 is not important as we dont feed it to the network 
        state_=torch.tensor(state_,dtype=torch.float32)
        
        # calculate the gradients 
        self.actor_critic.optimizer.zero_grad()
        state_value,probabilites=self.actor_critic(state)
        state_value_,_=self.actor_critic(state_) 
        
        state_value=torch.squeeze(state_value) # we squeeze the value to remove the extra dimension  [1,1] -> [1]
        state_value_=torch.squeeze(state_value_)
        
        action_probs=torch.distributions.Categorical(probabilites)
        log_prob=action_probs.log_prob(self.action) # we calculate the log probability of the action we took    
        
        delta = reward + self.gamma*state_value_*(1-int(done)) - state_value    # we calculate the TD error 
        actor_loss = -log_prob*delta    # we calculate the actor loss   
        critic_loss=delta**2       # we calculate the critic loss
        total_loss= actor_loss+critic_loss  
        
        total_loss.backward()  # we backpropagate the gradients
        self.actor_critic.optimizer.step()  # we update the weights of the network
        
        