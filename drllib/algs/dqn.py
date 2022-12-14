# from audioop import bias
# from os import NGROUPS_MAX
# from matplotlib import use
from collections import defaultdict
from torch import nn, optim
import torch
import numpy as np
import wandb

from drllib.utils.utils import get_conv2d_out_dim
from drllib.algs.core import BaseNNAlg
from drllib.utils.memory import ReplayMemory

class DQNPolicyCNN(nn.Module):
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
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=0),
            nn.BatchNorm2d(64),
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


class DQNPolicy(nn.Module):
    def __init__(
        self, 
        in_features: int,
        n_actions: int,
        lin_layers: list[int] = [64, 64],
        use_bn: bool = True,
        use_bias: bool = True
    ) -> None:
        
        super().__init__()

        net_layers = []
        layer_sizes = [in_features] + lin_layers

        for i in range(len(layer_sizes)-1):
            net_layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=use_bias),
                nn.ReLU(inplace=True)
            ])
            if use_bn:
                net_layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
        
        net_layers.append(nn.Linear(layer_sizes[-1], n_actions, bias=use_bias))

        self.policy = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.policy(x)

    def _get_action(self, obs):
        # perform greedy action here
        # perfrom exploration strategy in algorithm
        q_vals = self(obs)
        action = torch.argmax(q_vals, dim=1).reshape(-1)

        return action

POLICY_MAP = {
    "linear": DQNPolicy,
    "cnn": DQNPolicyCNN
}

class DQN(BaseNNAlg):

    def __init__(self, env, seed, policy, optim_kwargs, policy_kwargs, explore_kwargs, target_update=10, learning_starts=5000,
                 gamma=0.998, mem_size=int(1e6), batch_size=64, device="cpu", new_step_api=False, use_wandb=False
        ) -> None:

        super().__init__(env, seed, new_step_api)

        obs = env.reset()
        print("obs shape", obs)
        # obs_maxs = env.observation_space.high
        self.obs_maxs = np.array([1.5, 1.5, 5., 5., 3.14, 5., 1., 1. ], dtype=np.float32)
        print("obs high: ", self.obs_maxs)
        # init memory
        self.memory = ReplayMemory(max_len=int(mem_size), obs_shape=obs.shape, 
            act_shape=(1,), obs_maxs=self.obs_maxs, device=device)
        self.n_actions = self.env.action_space.n
        self.device = device
        self._train = True
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.eps_sched = self._setup_exploration(**explore_kwargs)
        self.use_wandb = use_wandb

        # init target q and policy q
        policy_type = POLICY_MAP.get(policy, DQNPolicy)
        self.q = policy_type(n_actions=self.n_actions, **policy_kwargs).to(device)
        self.target_q = policy_type(n_actions=self.n_actions, **policy_kwargs).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()
        self.q.train()
        self.target_update = target_update
        if self.use_wandb:
            # use all for weights and gradients
            wandb.watch(self.q, log="all", log_freq=50)

        # init loss + optimizer
        self.loss_fn = nn.SmoothL1Loss()
        # self.optimizer = optim.RMSprop(self.q.parameters(), **optim_kwargs)
        self.optimizer = optim.Adam(self.q.parameters(), **optim_kwargs)
    
    
    def train(self, n_iters):
        self.q.train()
        self._train = True
        return self._run_env(train=True, n_iters=n_iters, render=False)

    def test(self, render, n_iters=1):
        self.q.eval()
        self._train = False
    
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

                if render:
                    self.env.render(mode="human")

                ep_rew += rew
                steps += 1
                ep_len += 1
                
                obs = new_obs

            ep_rew_history.append(ep_rew)
            ep_len_history.append(ep_len)

            if i % 5 == 0: 
                print(f"Episode {i}: reward -> {ep_rew :.2f}, ep_len -> {ep_len}")
        return ep_rew_history, ep_len_history
    
    def _run_env(self, train, n_iters, render=False):
        
        steps = 0
        ep_rew_history = []
        ep_len_history = []
        # total_timesteps = 0

        # for every epoch
        for i in range(n_iters):

            obs = self.env.reset()
            done = False
            ep_rew = 0
            ep_len = 0
            ep_loss = []
            obs_data = defaultdict(list)
            curr_qvals_data = defaultdict(list)
            target_qvals_data = defaultdict(list)
            acts = []

            # TODO: switch to collecting rollouts (?) or at least switch to going multiple 
            # steps before sampling training

            while not done:
                
                # TODO: set policy to eval
                self.q.eval()
                tensor_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                # perform e-greedy action selection
                # print(tensor_obs.shape)
                a = self._get_action(tensor_obs)

                new_obs, rew, done, info = self._step_env(a)
                acts.append(a)
                # print("obs", )
                # TODO: is this where I clip rewards?? 
                # TODO: Add rew and obs normalization to memory.sample

                self.memory.append(obs, a, rew, new_obs, done)
                ep_rew += rew
                obs = new_obs

                warming_up = len(self.memory) < self.learning_starts
                
                # Sample minibatch
                # TODO set policy train to true
                self.q.train()
                if not warming_up and len(self.memory) >= self.batch_size:
                    # get batch transitions
                    b_tr = self.memory.sample(self.batch_size, to_tensor=True)

                    # for k, v in b_tr.items():
                    #     print(k, v.shape, v.dtype)

                    # print("qval out: ", self.target_q(b_tr["new_obs"]).shape)
                    with torch.no_grad():
                        next_q_vals, _ = torch.max(self.target_q(b_tr["new_obs"]), dim=1, keepdim=True)
                        # print("q val target: ", q_val.shape)

                        td_targets = b_tr["rew"] + self.gamma * next_q_vals * (1. - b_tr["done"])
                    
                    # print("actions: ", b_tr["act"].shape, b_tr["act"])
                    # select q val from action a
                    qvals = self.q(b_tr["obs"])
                    qvals_gather = torch.gather(qvals, dim=1, index=b_tr["act"].long())

                    obs_data['min'].append(torch.min(b_tr["obs"],dim=0)[0].detach().cpu().numpy())
                    obs_data['max'].append(torch.max(b_tr["obs"],dim=0)[0].detach().cpu().numpy())

                    curr_qvals_data['min'].append(torch.min(qvals_gather).item())
                    curr_qvals_data['max'].append(torch.max(qvals_gather).item())
                    curr_qvals_data['mean'].append(torch.mean(qvals_gather).item())

                    target_qvals_data['min'].append(torch.min(td_targets).item())
                    target_qvals_data['max'].append(torch.max(td_targets).item())
                    target_qvals_data['mean'].append(torch.mean(td_targets).item())

                    
                    # select using one hot vector - methods are the same
                    # act_select = nn.functional.one_hot(b_tr["act"].reshape(-1), num_classes=self.n_actions)
                    # print("qvals and action selection: ", act_select.shape, qvals.shape)
                    # qvals_select = torch.sum(act_select * qvals, dim=1, keepdim=True)
                    # print("gather: ", qvals_gather.shape)
                    # print("select: ", qvals_select.shape)
                    # print(torch.all(qvals_select == qvals_gather)) # is equal
                    # TODO: Add reward clipping!!
                    # print(td_targets.shape, qvals_gather.shape)
                    # print("current qvals: ", qvals_gather)
                    # print("target qvals: ", td_targets)

                    loss = self.loss_fn(td_targets, qvals_gather)
                    ep_loss.append(loss.item())
                    # print("loss: ", ep_loss[-1])

                    # backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    # clamp gradients between (-1, 1)
                    # for param in self.q.parameters():
                    #     nn.utils.clip_grad.clip_grad_value_(param, clip_value=1)
                        # param.grad.data.clamp_(-1,1)
                    nn.utils.clip_grad.clip_grad_norm_(self.q.parameters(), max_norm=10.)
                    self.optimizer.step()

                steps += 1
                ep_len += 1

                # Update targete every C steps
                if (not warming_up) and steps % self.target_update == 0:
                    self.target_q.load_state_dict(self.q.state_dict())
            
            # TODO: update epsilon
            if not warming_up:
                self.eps_sched.update()
            # self.eps = max(self.min_eps, self.eps * self.eps_decay)

            ep_rew_history.append(ep_rew)
            ep_len_history.append(ep_len)

            if i % 10 == 0: 
                print(f"Episode {i}: reward -> {ep_rew :.2f}, ep_len -> {ep_len}, ep_loss -> {np.mean(ep_loss):.2f}")
                # TODO: log statistics
                if self.use_wandb:
                    wandb.log({
                        "rollout/ep_reward":ep_rew,
                        "rollout/ep_rew_mean": np.mean(ep_rew_history[-100:]),
                        "rollout/ep_length": ep_len,
                        "rollout/ep_len_mean": np.mean(ep_len_history[-100:]),
                        "train/loss_mean": np.mean(ep_loss),
                        "rollout/exploration_rate": self.eps_sched.get_value(),
                        "global_step": steps,
                        "rollout/actions": wandb.Histogram(acts)
                    }, step=steps)

                    for k, v in obs_data.items():
                        feats = np.mean(np.stack(v, axis=0), axis=0)
                        # print("obs batch shape: ", feats.shape)
                        wandb.log({f"debug/obs_{i}_{k}" : feats[i] for i in range(len(feats))}, step=steps)


                    # wandb.log({f"debug/obs_{k}" : wandb.Histogram(np.mean(np.stack(v, axis=0), axis=0)) for k, v in obs_data.items()})
                    wandb.log({f"debug/curr_qvals_{k}" : np.mean(v) for k, v in curr_qvals_data.items()}, step=steps)
                    wandb.log({f"debug/target_qvals_{k}" : np.mean(v) for k, v in target_qvals_data.items()}, step=steps)

    def _get_action(self, obs):
        if self._train and (np.random.rand() < self.eps_sched.get_value()):
            return self.env.action_space.sample()

        # with torch.no_grad():
        a = self.q._get_action(obs).item()
        return a
        

if __name__ == "__main__": 
    # test network is working properly
    import torch
    from torchsummary import summary

    q_net = DQNPolicyCNN(n_actions=4, in_channels=4, img_size=84)

    b, h, d = 64, 84, 4
    summary(q_net, (d, h, h), batch_size=-1, device="cpu")
    input = torch.randn(b, d, h, h)
    print(input.shape)

    out = q_net(input)
    print(out.shape)

    q_net = DQNPolicy(in_features=10, n_actions=4, lin_layers=[128, 64])
    print(q_net)

    input = torch.randn(b, 10)
    print(input.shape)

    out = q_net(input)
    print(out.shape)