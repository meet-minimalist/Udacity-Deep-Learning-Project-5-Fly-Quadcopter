import numpy as np
from collections import deque

class memoryBuffer():
    def __init__(self, mem_size):
        self.memory = deque(maxlen = mem_size)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def retrive_mem(self, batch_size):
        idx = np.random.choice(np.range(len(self.memory)), size=batch_size, replace=False)
        return [self.memory[id] for id in idx]
    