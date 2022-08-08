import numpy as np
import random
import torch
from drllib.utils.schedulers import LinearScheduler, ExponentialScheduler


class BaseAlg(object):

    def __init__(self, env, policy, seed=None, new_step_api=False) -> None:
        # if deterministic, set seed
        self.env = env
        self.policy = policy
        self.new_step_api = new_step_api
        
        if seed is not None: self._set_seed(seed)

    def _set_seed(self, seed, seed_action_space=True):
        # Need to seed the env and action space if using random sampling
        # See https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        if seed_action_space:
            self.env.action_space.seed(seed)

    def _step_env(self, a):
        if self.new_step_api:
            # new step api return terminated or truncated instead of just done
            obs_, rew, term, trunc, info = self.env.step(a)
            done = term or trunc
        else:
            obs_, rew, done, info = self.env.step(a)
        return obs_, rew, done, info

    def train(self, n_iters):
        self.policy.train()
        return self._run_env(train=True, n_iters=n_iters, render=False)

    def test(self, render, n_iters=1):
        self.policy.eval()
        return self._run_env(train=False, n_iters=n_iters, render=render)

    def _run_env(self, train, n_iters, render=False):
        raise NotImplementedError

    def _setup_exploration(self, decay, val, end_val, duration):
        if decay == "linear":
            sched = LinearScheduler(val, end_val, duration=duration)
        else:
            sched = ExponentialScheduler(val, end_val, duration=duration)
        return sched
        

class BaseNNAlg(BaseAlg):
    
    def __init__(self, env, seed=None, new_step_api=False) -> None:
        super().__init__(env, None, seed, new_step_api)

    def _set_seed(self, seed):
        print("setting torch manual seed")
        torch.random.manual_seed(seed)
        super()._set_seed(seed)

    def train(self, n_iters):
        return self._run_env(train=True, n_iters=n_iters, render=False)


    def test(self, render, n_iters=1):
        return self._run_env(train=False, n_iters=n_iters, render=render)


    def _run_env(self, train, n_iters, render=False):
        raise NotImplementedError