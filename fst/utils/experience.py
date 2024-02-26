from collections import namedtuple, deque
import random
import numpy as np
import torch


Transition = namedtuple(
    "Transition",
    (
        "obs",
        "action",
        "n_obs",
        "reward",
        "terminated",
        "truncated",
        "joint_obs",
    )
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        clipped_batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, clipped_batch_size)

    @staticmethod
    def concat_batch(batch):
        return Transition(
            obs=np.concatenate(list(t.obs for t in batch)),
            action=np.concatenate(list(t.action for t in batch)),
            n_obs=np.concatenate(list(t.n_obs for t in batch)),
            reward=np.concatenate(list(t.reward for t in batch)),
            terminated=np.concatenate(list(t.terminated for t in batch)),
            truncated=np.concatenate(list(t.truncated for t in batch)),
            #joint_obs=np.concatenate(list(t.joint_obs for t in batch)),
            joint_obs=None,
            )

    @staticmethod
    def to_torch(transition):
        """Converts a transition with numpy arrays into one with torch tensors"""
        return Transition(
            obs=torch.FloatTensor(transition.obs),
            action=torch.LongTensor(transition.action),
            n_obs=torch.FloatTensor(transition.n_obs),
            reward=torch.FloatTensor(transition.reward),
            terminated=torch.BoolTensor(transition.terminated),
            truncated=torch.BoolTensor(transition.truncated),
            joint_obs=None,
            )

    def clear(self):
        self.memory.clear()

    def __iter__(self):
        return self.memory.__iter__()

    def __getitem__(self, key):
        return self.memory.__getitem__(key)

    def __setitem__(self, key, val):
        return self.memory.__setitem__(key, val)

    def __len__(self):
        return len(self.memory)
