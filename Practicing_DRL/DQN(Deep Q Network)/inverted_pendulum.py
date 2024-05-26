import os
import gym 
import torch 
import random
import numpy as np 
from torch.optim import Adam
import torch.nn as nn
import tensorboard
from stable_baselines3 import PPO,HER,A2C,DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy 

log_path=os.path.join('Training','Logs')
# Test random environment with openai gym 
print(log_path)
env = gym.make('CartPole-v0')
#env=DummyVecEnv([lambda:env])

states = env.observation_space.shape[0]
action = env.action_space.n 

episodes=10

for episode in range (1,episodes+1):
    state=env.reset()
    done=False
    score=0

    while not done:
        env.render()
        action=random.choice([0,1])
        next_state,reward,done,info=env.step(action)
        score+=reward

    print(f'Episode : {episode}  score: {score }')



# create a deep learning model with pytorch 

class Model(nn.Module):
    def init(self,state,action):
        super(Model,self).init()
        self.state=state
        self.action = action
        self.dense_layer= nn.Sequential(
            nn.Flatten(self.state,24),
            nn.ReLU(),
            nn.Flatten(24,24),
            nn.ReLU(),
            nn.Flatten(24,self.action),
        )
    def forward(self,x):
        return self.dense_layer(x)

model=Model(state,action)
print(model)

# bulid the agent 


ppo=PPO('MlpPolicy',env,verbose=1)
#ppo.compile(Adam(model.parameters(),lr=1e-3))
#ppo.fit(env,nb_steps=50000,visualize=2,verbose=1)

ppo.learn(total_timestep=20000)

