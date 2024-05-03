import numpy as np 
import gymnasium as gym
import time 
import random
import os 
from IPython.display import clear_output
from tqdm import tqdm


# Initalization 
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 

rewards_all_episodes = []

# Gym environment setup 
env = gym.make('FrozenLake-v1', render_mode='ansi')


# making the q table 
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

# Coding the Q-Learning Algorithm training loop


for episode in tqdm(range(num_episodes)):
    # first we have to reset the environment
    state=env.reset()[0]
    done=False
    rewards_current_episodes=0 #we start out with no rewards at the beginning of each episode.
        
    q_old=q_table
    # initalize the episode parametrs
    for step in range(max_steps_per_episode):
        #random=np.random.random()
        exploration_rate_treshold=random.uniform(0,1)
                
        # Exploration exploitation trade-off 
        if exploration_rate_treshold>exploration_rate:
            # get the higest value from the q table
            action=np.argmax(q_table[state,:])        
        else: 
            # choose action via exploration (randome value from the q table)
            action=env.action_space.sample()
        # take a new action 
        new_state,reward,done,truncated,info=env.step(action)
        
        # update the Q Table 
        q_table[state,action]=(1-learning_rate)*q_table[state,action]+\
            learning_rate*(reward + discount_rate*np.max(q_table[new_state,:]))
        
        # set new state (transition to new state)
        state=new_state
    
        # add new reward 
        rewards_current_episodes+=reward
        
        if done==True:
            break        
    # exploration rate decay
    
    exploration_rate=min_exploration_rate + (max_exploration_rate-min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    
    # add the current episode reward to total reward list 
    
    rewards_all_episodes.append(rewards_current_episodes)
    
    # calculate and print the average reward pre thouthand episods
    
rewards_per_thousand_episodes=np.split(np.array(rewards_all_episodes),num_episodes/1000)
count=1000
    
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)


########### Watch Q-Learning Agent Play Frozen Lake ##############


# Watch our agent play Frozen Lake by playing the best action 
# from each state according to the Q-table

for episode in range(1):
    # initialize new episode params
    state=env.reset()[0]
    done=False
    print("******* EPISODE ",episode+1 ,"*******\n\n\n")
    time.sleep(1) # sleep for one second

    for step in range(max_steps_per_episode):        
        # Show current state of environment on screen
        clear_output(wait=True)
        print(env.render())
        time.sleep(0.3)
        # Choose action with highest Q-value for current state    
        action=np.argmax(q_table[state,:])   
        # Take new action
        new_state,reward,done,trancated,info=env.step(action)

        if done:
            clear_output(wait=True)
            print(env.render())
            if reward == 1:
                # Agent reached the goal and won episode
                print("*********You reached the goal! ********")
                time.sleep(3)
            else:
                # Agent stepped in a hole and lost episode            
                print("****You fell through a hole!****")
                time.sleep(3)
                clear_output(wait=True)
            break
        # Set new state
        state=new_state
env.close()