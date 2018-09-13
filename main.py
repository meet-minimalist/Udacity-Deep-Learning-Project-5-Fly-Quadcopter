import sys
from agents.agent import agent
from task import Task
import numpy as np
import tensorflow as tf

num_episodes = 1000
# Target position = [10, 10, 10]
target_pos = np.array([10., 10., 10.])
agent = agent(targetPosition=target_pos)


#populate the memory of the agent with random experiences
#agent's memory can be accessed by agent.agentMemory()
agent.generate_memory()

with tf.Session() as sess:
    with i in range(1, num_episodes+1):
        reset_state = agent.reset_episode()
        
        # generate steps for few timesteps to add in memory
        
        # sample from the memory 
        
        # use the sample get action from agentNN providing it with current state
        
        # use current state and action from agentNN to get the Q value for current state and action pair from criticNN
        
        # use the Q value to update the weights of the actorNN and feed it with the nextState to get the nextAction based on updated actorNN
        
        # use the nextState and nextAction provided from the actorNN and supply it to criticNN to get the Q_nextState
        
        # use Q_nextState, Q_state, reward to have TD error and use the same to update the weights of criticNN
            
        for 
    

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()