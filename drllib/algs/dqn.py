from torch import nn, optim
import torch
import numpy as np
import wandb

from drllib.utils import get_conv2d_out_dim
from drllib.algs.core import BaseNNAlg
from drllib.utils.memory import ReplayMemory

class DQNPolicy(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        img_size: int,  
        n_actions: int,
    ) -> None:
        
        super().__init__()

        w = get_conv2d_out_dim(get_conv2d_out_dim(get_conv2d_out_dim(img_size, 8, 4), 4, 2), 3, 1)
        # print(f"Image width after cnn = {w}")
        # TODO: make easy way to add batch norm to network
        self.policy = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(8,8), stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(w*w*64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.policy(x)

    def _get_action(self, obs):
        # perform greedy action here
        # perfrom exploration strategy in algorithm
        q_vals = self(obs)
        action = torch.argmax(q_vals, dim=1).reshape(-1)

        return action



class DQN(BaseNNAlg):

    def __init__(self, env, seed, optim_kwargs, policy_kwargs, target_update=10, gamma=0.998,
                epsilon=0.5, min_eps=0.05, eps_decay=0.998, mem_size=int(1e6), device="cpu", new_step_api=False
        ) -> None:

        super().__init__(env, seed, new_step_api)

        obs = env.reset()
        # init memory
        self.memory = ReplayMemory(max_len=int(mem_size), obs_shape=obs.shape, act_shape=(1,))
        self.n_actions = self.env.action_space.n
        self.device = device
        self._train = True
        self.gamma = gamma
        self.eps = epsilon
        self.min_eps = min_eps
        self.eps_decay = eps_decay

        # init target q and policy q
        self.q = DQNPolicy(n_actions=self.n_actions, **policy_kwargs).to(device)
        self.target_q = DQNPolicy(n_actions=self.n_actions, **policy_kwargs).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()
        self.target_update = target_update
        wandb.watch(self.q, log_freq=50)

        # init loss + optimizer
        self.loss_fn = nn.SmoothL1Loss()
        # self.optimizer = optim.RMSprop(self.q.parameters(), **optim_kwargs)
        self.optimizer = optim.Adam(self.q.parameters(), **optim_kwargs)
    
    
    def train(self, n_iters, batch_size):
        return self._run_env(train=True, batch_size=batch_size, n_iters=n_iters, render=False)
    
    
    def _run_env(self, train, batch_size, n_iters, render=False):
        
        steps = 0
        ep_rew_history = []
        ep_len_history = []

        # for every epoch
        for i in range(n_iters):

            obs = self.env.reset()
            done = False
            ep_rew = 0
            ep_len = 0
            ep_loss = []

            while not done:
                
                tensor_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                # perform e-greedy action selection
                # print(tensor_obs.shape)
                a = self._get_action(tensor_obs)

                new_obs, rew, done, info = self._step_env(a)

                self.memory.append(obs, a, rew, new_obs, done)
                ep_rew += rew
                
                # Sample minibatch
                if len(self.memory) >= batch_size:
                    # get batch transitions
                    b_tr = self.memory.sample(batch_size, to_tensor=True)

                    # for k, v in b_tr.items():
                    #     print(k, v.shape, v.dtype)

                    # print("qval out: ", self.target_q(b_tr["new_obs"]).shape)
                    q_val, _ = torch.max(self.target_q(b_tr["new_obs"]), dim=1, keepdim=True)
                    # print("q val target: ", q_val.shape)

                    # with torch.no_grad():
                    td_targets = b_tr["rew"] + self.gamma * q_val * (1. - b_tr["done"])
                    
                    # print("actions: ", b_tr["act"].shape, b_tr["act"])
                    # select q val from action a
                    act_select = nn.functional.one_hot(b_tr["act"].reshape(-1), num_classes=self.n_actions)
                    qvals = self.q(b_tr["obs"])
                    # print("qvals and action selection: ", act_select.shape, qvals.shape)
                    qvals_select = torch.sum(act_select * qvals, dim=1, keepdim=True)

                    # print(td_targets.shape, qvals_select.shape)

                    loss = self.loss_fn(td_targets, qvals_select)
                    # print("loss: ", loss)
                    ep_loss.append(loss.item())

                    # backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                steps += 1
                ep_len += 1

                # Update targete every C steps
                if steps % self.target_update == 0:
                    self.target_q.load_state_dict(self.q.state_dict())
            
            # TODO: update epsilon
            self.eps = max(self.min_eps, self.eps * self.eps_decay)

            ep_rew_history.append(ep_rew)
            ep_len_history.append(ep_len)

            if i % 5 == 0: 
                print(f"Episode {i}: reward -> {ep_rew :.2f}, ep_len -> {ep_len}")
                # TODO: log statistics
                wandb.log({
                    "ep_reward":ep_rew,
                    "ep_reward_smooth": np.mean(ep_rew_history[-100:]),
                    "ep_length": ep_len,
                    "ep_length_smooth": np.mean(ep_len_history[-100:]),
                    "ep_mean_loss": np.mean(ep_loss),
                    "epsilon": self.eps,
                })

    def _get_action(self, obs):
        if self._train and (np.random.rand() < self.eps):
            return self.env.action_space.sample()

        return self.q._get_action(obs).item()
        

if __name__ == "__main__": 
    # test network is working properly
    import torch
    from torchsummary import summary

    q_net = DQNPolicy(n_actions=4, in_channels=4, img_size=84)

    b, h, d = 64, 84, 4
    summary(q_net, (d, h, h), batch_size=-1, device="cpu")
    input = torch.randn(b, d, h, h)
    print(input.shape)

    out = q_net(input)
    print(out.shape)