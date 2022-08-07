class BaseScheduler(object):
    def __init__(self, val, end_val, is_min=True):
        self.val = val
        self.end_val = end_val
        self.is_min = is_min

    def get_value(self):
        if self.is_min:
            self.val = max(self.val, self.end_val)
        else:
            self.val = min(self.val, self.end_val)
        return self.val
    
    def update(self):
        raise NotImplementedError


class LinearScheduler(BaseScheduler):

    def __init__(self, val, end_val, duration, is_min=True):
        super().__init__(val, end_val, is_min)
        self.step = (end_val - val)/duration

    def update(self):
        self.val += self.step

        return self.val


class ExponentialScheduler(BaseScheduler):

    def __init__(self, val, end_val, decay_rate=0.9998, duration=None, is_min=True):
        super().__init__(val, end_val, is_min)
        if duration:
            self.decay_rate = self._get_decay(duration)
        else:
            self.decay_rate = decay_rate
        print("Using decay rate: ", self.decay_rate)

    def _get_decay(self, duration):
        return (self.end_val/self.val)**(1/duration)

    def get_decay(self):
        return self.decay_rate
    
    def update(self):
        self.val *= self.decay_rate

        return self.val
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    duration = int(1e5)
    start, end = 1.0, 0.1
    decay = 0.9998
    lin = LinearScheduler(start, end, duration)
    exp = ExponentialScheduler(start, end, duration=duration)

    lin_vals = []
    exp_vals = []

    for i in range(duration):
        lin.update()
        exp.update()

        if i % 20 == 0:
            lin_vals.append(lin.get_value())
            exp_vals.append(exp.get_value())

    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    axs[0].plot(np.arange(0, duration, 20), lin_vals)
    axs[0].set_title(f"Linear")
    axs[1].plot(np.arange(0, duration, 20), exp_vals)
    axs[1].set_title(f"Exponential: decay rate {exp.get_decay():.6f}")
    plt.show()
