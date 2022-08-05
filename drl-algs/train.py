import gym
import json
import torch
import wandb

from wrappers.env_wrappers import MaxAndSkipEnv, ScaleFrame
from algs.dqn import DQN

def main(config_file):
    
    with open(config_file, "r") as f:
        config = json.load(f)
    print(config)

    to_render = True
    seed = 17
    mem_size = 1e6
    new_step_api = False
    epsilon = 0.5
    min_eps = 0.05
    n_iters = 100
    target_update=5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model on device {device}")

    wandb.init(project=config['env'], group="DQN-v0", config=config)
    config = wandb.config
    print("\nConfig:\n", config)

    env = gym.make(config['env'], **config['env_params'])
    # env = wrap_deepmind(en)v
    # try wrappers RecordVideo or RecordEpisodeStatistics to document episode stats

    env = MaxAndSkipEnv(env, frameskip=4, obs_len=4)
    env = ScaleFrame(env, 84, 84, 4)

    # Initialize agent
    dqn_agent = DQN(env, seed=seed, optim_kwargs=config["optim_params"], policy_kwargs=config["policy_params"], 
                    target_update=target_update, epsilon=epsilon, min_eps=min_eps, mem_size=mem_size, new_step_api=new_step_api)

    # Train agent
    # TODO: add save model
    dqn_agent.train(n_iters, 64)


if __name__ == "__main__":
    fn = "./configs/dqn-lunar-lander.json"
    main(fn)


