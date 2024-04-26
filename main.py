import os
import gym
import numpy as np
from Agent_Actor_Critic import Agent
from utils import plot_learning_curve

if __name__=='__main__':
    env=gym.make('CartPole-v1')
    agent=Agent(learning_rate=1e-5,n_actions=env.action_space)
    n_games=1800
    filename='cartpole.png'
    figure_file='plots/'+ filename
    
    best_score=env.reward_range[0]
    score_history=[]
    #load_checkpoint=False
    check_path_dir="tmp/actor_critic"
    if check_path_dir and os.path.exists(check_path_dir):
        
        agent.load_model()
        """ if load_checkpoint:
            agent.load_model()"""
    for i in range(n_games):
        observation=env.reset() # Resets the environment to an initial state and returns the initial observation.
        done=False
        score=0
        while not done:
            action=agent.choose_action(observation)
            observation_,reward,done,info =env.step()
            score+=reward
            
            if check_path_dir and os.path.exists(check_path_dir):
                agent.learn(observation,reward,observation_,done)
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
        
                
                
                
    
