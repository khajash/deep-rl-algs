import numpy as np
import random
import torch


class BaseAlg(object):

    def __init__(self, env, policy, seed, new_step_api=False) -> None:
        self._set_seed(seed)
        self.env = env
        self.policy = policy
        self.new_step_api = new_step_api

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def train(self, n_iters):
        self.policy.train()
        return self._run_env(train=True, n_iters=n_iters, render=False)

    def test(self, render, n_iters=1):
        self.policy.eval()
        return self._run_env(train=False, n_iters=n_iters, render=render)

    def _run_env(self, train, n_iters, render=False):
        raise NotImplementedError