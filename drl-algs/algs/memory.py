# from collections import namedtuple, deque
import random
from typing import List
import numpy as np
import torch

class CircularBuffer(object):
    """Circular buffer with max size for storing a fixed amount of data. After reaching the 
        end of buffer, starts over at index 0 and writes over old data

        Args:
            max_len (int): _description_
            data_dims (List[int]): _description_
        
    """
    def __init__(
        self, 
        max_len: int, 
        data_dims: tuple,
        dtype: str = 'float32'
    ) -> None:

        if isinstance(data_dims, int):
            data_dims = (data_dims,)

        self.data_buff = np.zeros((max_len,) + data_dims, dtype=dtype)
        self.max_len = max_len
        self.insert_idx = 0
        self.is_full = False

    def __len__(self) -> int:
        if self.is_full:
            return self.max_len
        else:
            return self.insert_idx

    def append(self, data: np.ndarray) -> None:
        self.data_buff[self.insert_idx] = data
        
        # Adjust insertion index and check if buffer is full
        self.insert_idx = self.insert_idx + 1
        if self.insert_idx >= self.max_len:
            self.is_full = True
            self.insert_idx = self.insert_idx % self.max_len

    # TODO: baselines uses changing start idx - understand why
    def get_batch(self, batch_idxs: np.ndarray) -> np.ndarray:
        return self.data_buff[batch_idxs]
    

class ReplayMemory(object):
    def __init__(
        self, 
        max_len: int,
        obs_shape: tuple,
        act_shape: tuple,
        device: str = 'cpu',
        dtype: str = 'float32'
    ) -> None:

        self.device = device

        self.obs = CircularBuffer(max_len, obs_shape, dtype)
        self.act = CircularBuffer(max_len, act_shape, 'int')
        self.rew = CircularBuffer(max_len, (1,), dtype)
        self.new_obs = CircularBuffer(max_len, obs_shape, dtype)
        self.done = CircularBuffer(max_len, (1,), dtype)

    def sample(self, batch_size: int, to_tensor: bool = False) -> dict:
        if batch_size > len(self.obs):
            raise IndexError("Searching for one or more index out of range")
            
        batch_idxs = np.random.choice(len(self.obs), batch_size, replace=False)
        batch_obs = self.obs.get_batch(batch_idxs)
        batch_act = self.act.get_batch(batch_idxs)
        batch_rew = self.rew.get_batch(batch_idxs)
        batch_new_obs = self.new_obs.get_batch(batch_idxs)
        batch_done = self.done.get_batch(batch_idxs)

        # TODO: baselines expands 1D arrays to (N,1) arrays - this is probably for loss or other calcs later
        batch = {
            "obs" : batch_obs,
            "act" : batch_act,
            "rew" : batch_rew,
            "new_obs" : batch_new_obs,
            "done" : batch_done
        }

        if to_tensor:
            # TODO: somewhere need to convert channels from (H, W, C) -> (C, H, W) for torch
            for k, v in batch.items():
                batch[k] = torch.from_numpy(v).to(device=self.device)

        return batch

    def append(
        self, 
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        new_obs: np.ndarray,
        done: np.ndarray
    ) -> None:
        
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.new_obs.append(new_obs)
        self.done.append(done)

    def __len__(self):
        return len(self.obs)


if __name__ == "__main__":
    # Test Replay Memory is working properly
    def gen_fake_transition(obs_shape, act_shape):
        tr = {
            "obs" : np.random.normal(size=obs_shape),
            "act" : np.random.normal(size=act_shape),
            "rew" : np.random.normal(size=(1,)),
            "new_obs" : np.random.normal(size=obs_shape),
            "done" : np.random.choice(2, 1)
        }
        return tr
    
    obs_shape =  (84, 84, 4)
    act_shape = (5,)
    memory = ReplayMemory(100, obs_shape, act_shape)
    batch_size = 64

    for i in range(30):
        tr = gen_fake_transition(obs_shape, act_shape)
        memory.append(**tr)

    print(f"Memory 1: {len(memory)}")

    # TODO: make sure I'm using errors correctly
    try: 
        memory.sample(batch_size)
    except IndexError as e:
        print("Error: ", e)

    for i in range(80):
        tr = gen_fake_transition(obs_shape, act_shape)
        memory.append(**tr)

    print(f"Memory 2: {len(memory)}")

    batch = memory.sample(batch_size)
    for k, v in batch.items():
        print(k, v.shape)
    


