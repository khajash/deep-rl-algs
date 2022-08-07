import numpy as np
import random
import torch


class BaseAlg(object):

    def __init__(self, env, policy, seed=None, new_step_api=False) -> None:
        # if deterministic, set seed
        if seed: self._set_seed(seed)
        self.env = env
        self.policy = policy
        self.new_step_api = new_step_api

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        # TODO: env seed not working
        # self.env.seed(seed)

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