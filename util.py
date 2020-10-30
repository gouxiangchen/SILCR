import heapq
import random
import numpy as np
from collections import deque


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path):
        b = np.load(path+'.npy', allow_pickle=True)
        assert(b.shape[0] == self.memory_size)

        for i in range(b.shape[0]):
            self.add(b[i])
        print('expert replay loaded!')


class Transition:
    def __init__(self, episode_reward, experience):
        self.data = (episode_reward, experience)

    def __lt__(self, other):
        return self.data[0] < other.data[0]


class PriorityMemory:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.buffer = []
        
    def add(self, experience, episode_reward):
        transition = Transition(episode_reward, experience)
        if len(self.buffer) == self.memory_size:
            lowest = heapq.heappop(self.buffer)
            # print('lowest episode reward : ', lowest.data[0])
            if lowest.data[0] < episode_reward:
                heapq.heappush(self.buffer, transition)
            else:
                heapq.heappush(self.buffer, lowest)
        else:
            heapq.heappush(self.buffer, transition)
    
    def get_lowest_rewards(self):
        lowest = heapq.heappop(self.buffer)
        heapq.heappush(self.buffer, lowest)
        return lowest.data[0]

    def sample(self, batch_size, continuous=False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i].data[1] for i in indexes]

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

