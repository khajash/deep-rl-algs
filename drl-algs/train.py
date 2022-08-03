from textwrap import wrap
import gym
from wrappers.atari_wrappers import wrap_deepmind
import json
import matplotlib.pyplot as plt
from wrappers.env_wrappers import MaxAndSkipEnv, ScaleFrame

def main(config_file):
    # with open(config_file, "r") as f:
    #     config = json.load(f)
    # print(config)

    # env = gym.make(config['env'], **config['env_params'])
    # env = wrap_deepmind(en)v
    # try wrappers RecordVideo or RecordEpisodeStatistics to document episode stats


    env = gym.make(
        "LunarLander-v2",
        continuous = False, # makes action space continuous
        gravity = -10.0,
        enable_wind = False,
        wind_power = 15.0,
        turbulence_power = 1.5,
        # render_mode = "rgb_array"
    )

    to_render = True

    env = MaxAndSkipEnv(env, frameskip=4, obs_len=4)
    env = ScaleFrame(env, 84, 84, 4)
    # env = gym.make( "LunarLander-v2", continuous = False, gravity = -10.0, enable_wind = False, wind_power = 15.0, turbulence_power = 1.5)

    obs = env.reset()
    done = False

    print("obs shape", obs.shape, obs.min(), obs.max())
    # plt.imshow(obs)
    # plt.show()
    i = 0
    while not done:
        action = env.action_space.sample()

        out = env.step(action)
        # print
        if len(out) == 4:
            # print("old step")
            new_obs, rew, done, info = out
        else: 
            new_obs, rew, term, trunc, info = out
            done = term or trunc
        # print("done:", done)
        print("obs here: ", new_obs.shape, new_obs.min(), new_obs.max())
        if to_render:
            env.render(mode="human")

        i += 1
        if i > 5:
            break
    env.close()

if __name__ == "__main__":
    fn = "./configs/dqn-atari.json"
    main(fn)


