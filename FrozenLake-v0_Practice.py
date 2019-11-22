#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import pandas as pd
import numpy as np
import random
from time import sleep
from itertools import combinations
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import clear_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


random.seed(100)
env = gym.make('FrozenLake-v0')
env = env.unwrapped


# In[3]:


env.render()


# Action - 0: Left, 1: Down, 2: Right, 3: Up

# In[4]:


state = env.observation_space.n
action = env.action_space.n
print((state, action))


# In[5]:


q_table_actual = np.zeros([state, action])
q_table_actual


# In[6]:


total_episodes = 10000
total_test_episodes = 10
max_steps = 300

learning_rate = [0.1]
gamma = [0.99]
learning_rate, gamma = np.meshgrid(learning_rate, gamma)
combinations = np.c_[learning_rate.ravel(),gamma.ravel()]

epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.001


# In[7]:


final_result = []
check = []
#tune learning rate and gamma
for lr, dr in combinations:
    state = env.observation_space.n
    action = env.action_space.n
    q_table = np.zeros([state, action])
    epsilon = 1
    #to check if the q_table is empty
    display(q_table)
    reward_for_all_episodes = []

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        for step in range(max_steps):
            random_value = random.uniform(0,1)
            if random_value > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)

            #capture actual reward
            total_reward += reward
            #update q_table
            q_table[state, action] = q_table[state, action] +                                     lr * (reward + dr * (np.max(q_table[next_state]))- q_table[state, action])
            step +=1
            state = next_state

            if done:
                break
#                 print(f'Episode: {episode}, Step: {step}, Reward: {reward} Done: {done}')
        #update epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        #append each episode's reward to a list
        reward_for_all_episodes.append(total_reward)

    average_reward = np.split(np.array(reward_for_all_episodes),total_episodes/1000)
    count = 0
    print(f'\nLearning rate: {lr}, Discount Rate: {dr}\n')
    for i in average_reward:
        count += i.shape[0]
        print(f'{count}: {sum(i) / i.shape[0]}')
    final_result.extend([[episode, lr, dr, sum(reward_for_all_episodes)]])

#     compare which combination of learning rate and gamma has the largest total reward,
#     save the final q_table to q_table_actual
    if check == []:
        check = [sum(reward_for_all_episodes)]
        q_table_actual = q_table
    else:
        if sum(reward_for_all_episodes) > check[0]:
            check = [sum(reward_for_all_episodes)]
            q_table_actual = q_table


# In[8]:


#save the result to a dataframe
final_result_df = pd.DataFrame(final_result, columns = ['Episode', 'Learning Rate', 'Discount Rate', 'Total Rewards'])


# In[9]:


#best combination
final_result_df[final_result_df['Total Rewards'] == final_result_df['Total Rewards'].max()]


# In[10]:


#plot graph
final_result_df['Combination'] =  list(zip(final_result_df['Learning Rate'], final_result_df['Discount Rate']))
plt.figure(figsize = (10,8))
ax = plt.axes()
ax = sns.barplot(final_result_df['Combination'], final_result_df['Total Rewards'])
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.show()


# In[11]:


check


# In[12]:


q_dataframe = pd.DataFrame(q_table_actual)
q_dataframe.style.applymap(lambda x:'background-color:yellow' if x != 0 else '')


# In[13]:


q_table


# In[14]:


#play the game
get_ipython().run_line_magic('time', '')
Win = 0
Lose = 0
average_step = []
for episode in range(total_test_episodes):
    state = env.reset()
    done = False
    step_episode = 0
    for step in range(max_steps):
        action = np.argmax(q_table_actual[state])
        next_state, reward, done, info = env.step(action)
        clear_output(wait = True)
        print(f'Game: {episode + 1}, Step: {step}, Reward: {reward}\n')
        print(env.render(mode = 'ansi'))
        sleep(0.5)
        state = next_state
        step_episode += 1
        if done:
            if reward == 1:
                Win +=1
                average_step.append(step_episode)
            else:
                Lose +=1
            break
    sleep(2)
print(f'Total Game: {total_test_episodes}\nWin: {Win}\nLose: {Lose}\nAverage Steps: {sum(average_step)/Win}')


# In[ ]:




