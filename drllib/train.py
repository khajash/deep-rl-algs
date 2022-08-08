import gym
import json
import torch
import wandb

from wrappers.env_wrappers import MaxAndSkipEnv, ScaleFrame
from algs.dqn import DQN

def setup_training_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/dqn-lunar-lander.json", 
        type=str,
        help="Save model when done.",
    )
    parser.add_argument(
        "--seed",
        default=17,
        type=int,
        help="Random seed. (int, default = 17)",
    )
    parser.add_argument(
        "--n_epochs",
        default=75,
        type=int,
        help="Number of epochs to run the training. (int, default = 75)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Do not initialize wandb logger"
    )
    parser.add_argument(
        "--new_step_api",
        action="store_true",
        help="Use new step API in OpenAI env (new_obs, rew, term, trunc, info) vs (new_obs, rew, done, info)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render env in testing"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model when done.",
    )

    return parser.parse_args()

def main():

    args = setup_training_parser()
    cl_config = vars(args)

    
    with open(args.config, "r") as f:
        default_config = json.load(f)
    print(default_config)

    # add notes, tags, 
    default_config.update(**cl_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model on device {device}")

    wandb.init(project=default_config['env'], group=f"{default_config['network']}-v0", config=default_config)
    config = wandb.config
    print("\nConfig:\n", config)

    env = gym.make(config['env'], **config['env_params'])

    # Add custom env wrappers
    # try wrappers RecordVideo or RecordEpisodeStatistics to document episode stats
    env = MaxAndSkipEnv(env, frameskip=4, obs_len=4)
    env = ScaleFrame(env, 84, 84, 4)

    # Initialize agent
    dqn_agent = DQN(env, seed=config["seed"], optim_kwargs=config["optim_params"], policy_kwargs=config["policy_params"], 
                    new_step_api=config["new_step_api"], device=device, **config["alg_params"])

    # Train agent
    # TODO: add save model
    dqn_agent.train(config["n_epochs"])


if __name__ == "__main__":
    main()


