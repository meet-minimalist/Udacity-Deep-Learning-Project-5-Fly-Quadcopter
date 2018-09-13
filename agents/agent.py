import numpy as np
import random
from task import Task
from memoryBuffer import memoryBuffer

class agent():
    def __init__(self, targetPosition=None):
        if targetPosition is None:
            self.target_pos = np.array(np.random.randint(0, 100, size=3))
        else:
            self.target_pos = targetPosition
            
    def act_random(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]
    
    def generate_memory(self, rand_sample_iteration=100, memorySize=20000):
        for i in range(rand_sample_iteration):
            # Pose / state contains positional data [x, y, z, phi, theta, psi]
            self.init_pose = np.array(np.random.randint(0, 100, size=6))
            self.init_velocities = np.array(np.random.uniform(0, 10, size=3))
            self.init_angle_velocities = np.array(np.random.uniform(0, 10, size=3))
            self.target_pos_for_exp = np.array(np.random.randint(0, 100, size=3))
         
            myTask = Task(self.init_pose, self.init_velocities, self.init_angle_velocities, 10, self.target_pos_for_exp)
            self.agentMemory = memoryBuffer(mem_size=memorySize)
            
            timeSteps = 200
            state = self.init_pose
            for i in range(timeSteps):
                rotor_speeds = agent.act_random(self)
                next_state, reward, done = myTask.step(rotor_speeds)
                # You will get 3 sets of outputs representing 3 timesteps
                # input rotor speed --> next-state-1 --> apply the same rotor speed --> next-state-2 --> apply the same rotor speed --> next-state-3
                # combine all the 3 states and represent it in next_state value
                # each next_state contains positional data [x, y, z, phi, theta, psi]
                # reward is also the sum of individual action rewards
                # done indicates whether we reached the target or not
                
                # Storing the experience
                # State(Pose-6 values), Action(Rotor speed-4 values), Reward, Next_State(Pose-6 values)
                for i in range(int(len(next_state)/6)):
                    self.agentMemory.add([state, rotor_speeds, reward[i], next_state[i: i+6]])
                    state = next_state[i: i+6]
#                if done == True:
                    #break
                
    def reset_episode(self):
        self.init_pose = np.array(np.random.randint(0, 100, size=6))
        self.init_velocities = np.array(np.random.uniform(0, 10, size=3))
        self.init_angle_velocities = np.array(np.random.uniform(0, 10, size=3))
        self.total_reward = 0
        self.count = 0
            
a = agent()
a.generate_memory()
b = a.agentMemory.memory
print(len(b))



