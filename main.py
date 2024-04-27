import os
import gym
import numpy as np
from Agent_Actor_Critic import Agent
import matplotlib.pyplot as plt
from utils import plot_learning_curve

if __name__=='__main__':
    env=gym.make('CartPole-v1')
    agent=Agent(learning_rate=1e-5,gamma=0.99,n_actions=env.action_space.n)
    n_games=1400
    filename='cartpole.png'
    figure_file='plots/'+ filename
    
    best_score=env.reward_range[0]
    score_history=[]
    #load_checkpoint=False
    check_path_dir="tmp/actor_critic"
    
    dir_name = 'D:\\MARS\\IRT_EXUPERY\\Actual Internship\\CODE\\Duckiebot_From-SIM-to-Real\\plots\\'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    if check_path_dir and os.path.exists(check_path_dir):
        
        agent.load_model()
        """ if load_checkpoint:
            agent.load_model()"""
    for i in range(n_games):
        observation=env.reset() # Resets the environment to an initial state and returns the initial observation.
        observation=observation[0] # extracting observation_array
        #print("Initial observation shape:", observation)
        done=False
        score=0
        while not done:
            action=agent.choose_action(observation)
            #print(env.step(action))
            observation_, reward, done, _, info =env.step(action)
            #print("Next observation shape:", observation_.shape) 
            score+=reward
            agent.learn(observation,reward,observation_,done)
            
            """ if check_path_dir and os.path.exists(check_path_dir):
                agent.learn(observation,reward,observation_,done)"""
            observation=observation_
        score_history.append(score)
        avg_score=np.sum(score_history[-100:])
        
        if avg_score>best_score:
            best_score=avg_score
        if not check_path_dir and os.path.exists(check_path_dir):
            agent.save_model()
        print("episode",i,"score %.1f" %score,"avg_score %.1f" %avg_score)
    x=[i+1 for i in range(n_games)]
    plot_learning_curve(x,score_history,figure_file)    
        
                
                
                
    
