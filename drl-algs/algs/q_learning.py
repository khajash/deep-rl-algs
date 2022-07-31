import numpy as np
import random
from tqdm import tqdm
from .core import BaseAlg 

class QLearning(BaseAlg):

    def __init__(self, env, policy, seed, new_step_api=False) -> None:
        super().__init__(env, policy, seed, new_step_api)
    
    def _run_env(self, train, n_iters, render=False):
        
        rew_list = []
        ep_len = []
        for i in tqdm(range(n_iters)):
            done = False
            obs= self.env.reset()
            total_r = 0
            e = 0
            while not done:
                # select action
                a = self.policy.act(obs, i)

                # take action
                if self.new_step_api:
                    # new step api return terminated or truncated instead of just done
                    obs_, rew, term, trunc, _ = self.env.step(a)
                    done = term or trunc
                else:
                    obs_, rew, done, _ = self.env.step(a)
                
                
                # render env
                if render:
                    self.env.render()
                
                # update policy
                if train:
                    self.policy.update(obs, a, rew, obs_)
                
                # update metrics
                e +=1
                total_r += rew
                obs = obs_
                if done: break
            if i % 100 == 0:
                print(f"Episode {i}: \tepisode len: {e} \tepisode_rew: {total_r}")
            rew_list.append(total_r)
            ep_len.append(e)
        return rew_list, ep_len


class QTable(object):
    def __init__(self, env, lr, gamma, eps=0.1) -> None:
        # add support for continuous envs - discretize env - use wrapper here rather than change the env
        self.n_obs = env.observation_space.n
        self.n_act = env.action_space.n
        print(f"Actions {self.n_act}, Obs {self.n_obs}")
        # self.actions = list(range(self.n_act))
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        # self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.reset()
    
    def reset(self):
        self.q_table = np.zeros([self.n_obs, self.n_act])
    
    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False
    
    def act(self, obs, i):
        # epsilon-greedy
        if self._train and (random.random() < self.eps):
        # if self._train:
            a = np.random.choice(self.n_act, 1)
            # a = np.argmax(self.q_table[obs] + np.random.randn(1,self.n_act)*(1./(i+1)))
        else:
            a = np.argmax(self.q_table[obs])
        return int(a)

    def update(self, obs, a, rew, obs_):
        self.q_table[obs, a] += self.lr * (rew + self.gamma * (self.q_table[obs_, np.argmax(self.q_table[obs_])]) - self.q_table[obs, a])



if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    new_step_api = True
    lr = 0.8
    gamma = 0.95
    eps = 0.3
    n_iters = 10000
    seed = 0

    env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=True, new_step_api=new_step_api)
    policy = QTable(env, lr, gamma, eps)
    agent = QLearning(env, policy, seed, new_step_api)

    rew_list, ep_len = agent.train(n_iters)

    # plot stats
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(rew_list)
    axs[1].plot(ep_len)
    plt.show()

    agent.test(True, 2)

    