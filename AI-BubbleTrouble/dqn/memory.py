import random
from collections import deque
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def push(self, experience):
        """
        Push the given experience to the memory buffer
        :param experience: Experience object contains ('state', 'action', 'next_state', 'reward', 'done')
        """
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        """
        :param priority_scale: scale factor is a number in [0,1]
        :return: The current sample probabilities
        """
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        """
        :param probabilities: sample probabilities
        :return: The current importance
        """
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        """
        :param batch_size: Number of samples
        :param priority_scale: scale factor is a number in [0,1]
        :return: list of samples in size of the given batch_size, there importance
        and there indices in the memory buffer
        """
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = [self.buffer[i] for i in sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        """
         Sets new priorities to the experiences which there indices are given
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def can_provide_sample(self, batch_size):
        """
        :return: True if there enough samples to sample and false otherwise
        """
        return len(self.buffer) >= batch_size


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        """
        Push the given experience to the memory buffer
        :param experience: Experience object contains ('state', 'action', 'next_state', 'reward', 'done')
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        """
        :param batch_size: Number of samples
        :return: list of samples in size of the given batch size
        """
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        """
        :return: True if there enough samples to sample and false otherwise
        """
        return len(self.memory) >= batch_size
