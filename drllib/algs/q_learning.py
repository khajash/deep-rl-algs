import numpy as np
import random
from tqdm import tqdm
from gym.spaces import Discrete

from drllib.algs.core import BaseAlg


class QLearning(BaseAlg):

    def __init__(self, env, policy, seed, new_step_api=False) -> None:
        super().__init__(env, policy, seed, new_step_api)
    
    def _run_env(self, train, n_iters, render=False):
        
        rew_list = []
        rew_avgd_list = []
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
                    self.policy.update(obs, a, rew, obs_, done)
                
                # update metrics
                e +=1
                total_r += rew
                obs = obs_

            # Decay lr and epsilon noise - (go from exploration to exploitation)
            self.policy.update_lr()
            self.policy.update_eps()
            
            if i % 100 == 0:
                print(f"Episode {i}: \tepisode len: {e} \tepisode_rew: {total_r}")
            
            rew_list.append(total_r)
            rew_avgd_list.append(np.mean(rew_list[-10:]))
            ep_len.append(e)


        return rew_avgd_list, ep_len


class QTable(object):
    def __init__(self, env, gamma, lr, lr_decay=0.998, eps=0.1, eps_final=0.05, warmup=800) -> None:
        # Currently only works with discrete action and observation spaces
        # TODO: add support for continuous envs - discretize env - use wrapper here rather than change the env
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)
        
        self.n_obs = env.observation_space.n
        self.n_act = env.action_space.n
        print(f"Actions {self.n_act}, Obs {self.n_obs}")

        self.env = env
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.eps = eps
        self.eps_final = eps_final
        self.eps_step = (eps - eps_final) / warmup

        self.reset()
    
    def reset(self):
        self.q_table = np.zeros([self.n_obs, self.n_act])
    
    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False
    
    def update_eps(self):
        self.eps = max(self.eps_final, self.eps - self.eps_step)
    
    def update_lr(self):
        self.lr *= self.lr_decay
    
    def get_policy(self): 
        return self.q_table
    
    def act(self, obs, i):
        # epsilon-greedy
        if self._train and (np.random.rand() < self.eps):
            return self.env.action_space.sample()
        
        # In case multiple actions have the same Q-value, especially at the beginning when all Q-values may be 0
        # Select all best actions, then select one randomly from best
        max_q = np.max(self.q_table[obs])
        best_acts = [a for a, q in enumerate(self.q_table[obs]) if q == max_q ]

        return np.random.choice(best_acts)


    def update(self, obs, a, rew, obs_, done):
        if done:
            td_target = rew
        else:
            td_target = rew + self.gamma * np.max(self.q_table[obs_, :])
        
        self.q_table[obs, a] += self.lr * (td_target - self.q_table[obs, a])



if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # set parameters
    new_step_api = True
    lr = 1.0
    lr_decay = 0.998
    gamma = 0.95
    eps = 1
    n_iters = 5000
    warmup = 4000
    seed = 0

    env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=True, new_step_api=new_step_api)
    policy = QTable(env, gamma, lr, lr_decay, eps, warmup=warmup)
    agent = QLearning(env, policy, seed, new_step_api)

    rew_list, ep_len = agent.train(n_iters)

    # generate a table with the optimal action displayed
    qtable = policy.get_policy()
    optimal_policy = np.argmax(qtable, axis=1).reshape(4,4)

    # plot stats
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(rew_list)
    axs[1].plot(ep_len)

    # plot grid of optimal action
    im = axs[2].imshow(optimal_policy, interpolation=None)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.show()

    # test agent following optimal policy
    agent.test(render=True, n_iters=1)

    