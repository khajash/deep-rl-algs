import gym
import json

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback
import torch



def setup_training_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/dqn-lunar-lander-linear.json", 
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
        default=10000,
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

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model on device {device}")

    use_wandb = not args.disable_wandb

    if use_wandb:
        run = wandb.init(
            project=config['env'], 
            group=f"{config['network']}-stable-v0", 
            config=config, 
            sync_tensorboard=True, 
            monitor_gym=True, 
            save_code=True
        )

    print("\nConfig:\n", config)

    # env = gym.make(config['env'], **config['env_params'])
    def make_env():
        env = gym.make(config['env'])
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    # model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    # model.learn(
    #     total_timesteps=config["total_timesteps"],
    #     callback=WandbCallback(
    #         gradient_save_freq=100,
    #         model_save_path=f"models/{run.id}",
    #         verbose=2,
    #     ),
    # )
    # model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    env = make_env()
    alg_config = config['alg_params']
    model = DQN("MlpPolicy", env, verbose=1, 
        learning_rate=config['optim_params']['lr'], 
        buffer_size=int(alg_config['mem_size']),
        learning_starts=alg_config['learning_starts'],
        batch_size=alg_config['batch_size'],
        gamma=alg_config['gamma'],
        tensorboard_log=f"runs/{run.id}",
        exploration_fraction=0.8, 
    )


    model.learn(
        total_timesteps=10000000, 
        log_interval=4, 
        callback=WandbCallback(
            gradient_save_freq=50,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    model.save("dqn_lunar_lander")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn_lunar_lander")

    obs = env.reset()
    for i in range(5):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                break
    env.close()

if __name__ == "__main__":
    main()