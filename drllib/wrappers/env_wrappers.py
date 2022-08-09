import gym
from collections import deque
import numpy as np
from gym import spaces
import cv2

# TODO: Look into separating skip and max step
# TODO: Separate image editing

class MaxAndSkipEnv(gym.Wrapper):
    # gym.wrapper for skip frames and calculating max of two images and stacking all in one
    def __init__(self, env=None, frameskip=4, obs_len=4) -> None:
        super().__init__(env)
        self._frameskip = frameskip
        self.obs_len = obs_len
        self._img_buffer = deque(maxlen=2)
        self._obs_buffer = deque([], maxlen=obs_len)

    def step(self, action):
        total_rew = 0.0
        done = False
        
        # print(f"starting skip step obs buffer {len(self._img_buffer)}")

        for i in range(self._frameskip):
            # TODO: if self.new_step_api:
            _, rew, done, info = self.env.step(action)
            img = self.env.render("rgb_array")
            self._process_img_data(img)
            total_rew += rew
            if done:
                break
            
        new_obs = np.stack(self._obs_buffer, axis=2)

        # TODO: account for if done before large buffer - may happen if frameskip is lower than . 
        # print(new_obs.shape)
        if new_obs.shape[-1] < self.obs_len:
            print("extending obs")
            n = self.obs_len - new_obs.shape[-1]
            new_obs = np.concatenate([new_obs]+ [np.expand_dims(new_obs[...,2], axis=-1)]*n, axis=-1)

        assert new_obs.shape[-1] == self.obs_len

        return new_obs, rew, done, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._img_buffer.clear()
        self._obs_buffer.clear()
        _ = self.env.reset(**kwargs)
        img = self.env.render("rgb_array")
        self._img_buffer.append(img)

        # obs needs a stack, so step environment and create stack
        for _ in range(self._frameskip):
            self.env.step(0) # noop
            img = self.env.render("rgb_array")
            self._process_img_data(img)
        new_obs = np.stack(self._obs_buffer, axis=2)
        assert new_obs.shape[-1] == self.obs_len
        
        return new_obs

    def _process_img_data(self, img):
        self._img_buffer.append(img)
        if len(self._img_buffer) > 1:
            max_img = _max_per_channel(self._img_buffer)
            y_img =  _rgb_to_y(max_img, from_uint8=True)
            self._obs_buffer.append(y_img)


class ScaleFrame(gym.Wrapper):
    # gym.wrapper for scaling frames
    def __init__(self, env, height, width, channels, interpolation=cv2.INTER_LINEAR) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(channels, height, width), dtype=np.float32)
        self._img_out = (height, width)
        self.inter = interpolation
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        scaled_obs = cv2.resize(obs, dsize=self._img_out, interpolation=self.inter)
        scaled_obs = np.transpose(scaled_obs, (2, 0, 1))
        return scaled_obs, rew, done, info 

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return np.transpose(cv2.resize(obs, dsize=self._img_out, interpolation=self.inter), (2, 0, 1))



def _max_per_channel(stack):
    img = np.zeros_like(stack[0])
    num_channels = stack[0].shape[-1]
    for i in range(num_channels):
        img[..., i] = np.max(np.stack([stack[0][..., i], stack[1][..., i]], axis=-1), axis=-1)
    return img


def _rgb_to_y(rgb_img: np.ndarray, from_uint8=True) -> np.ndarray:
    # convert rgb to luma channel
    if from_uint8:
        rgb_img = rgb_img.astype(np.float32) / 255.
    y = 0.299 * rgb_img[...,0] + 0.587 * rgb_img[...,1] + 0.114 * rgb_img[...,2]
    return y