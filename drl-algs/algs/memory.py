from collections import namedtuple, deque
import random
from typing import List

Transition = namedtuple(
    "Transition", 
    ("obs", "action", "rew", "new_obs")
    # ("obs", "action", "rew", "new_obs", "done") 
)

# TODO: check if there is something more efficient
class ReplayMemory(object):
    def __init__(self, max_len: int) -> None:
        self.memory = deque([], max_len=max_len)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, k=batch_size)

    def add(self, *args) -> None:
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)