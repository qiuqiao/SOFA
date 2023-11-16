import numpy as np


class GaussianRampUpScheduler:
    def __init__(self, max_steps, start_steps=None, end_steps=None):
        if end_steps is None:
            end_steps = max_steps
        if start_steps is None:
            start_steps = 0
        self.max_steps = max_steps
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.curr_steps = 0

    def __call__(self):
        if self.curr_steps < self.start_steps:
            return 0
        elif self.curr_steps < self.end_steps:
            return np.exp(
                -5
                * (
                    1
                    - (self.curr_steps - self.start_steps)
                    / (self.end_steps - self.start_steps)
                )
                ** 2
            )
        else:
            return 1

    def step(self):
        self.curr_steps += 1

    def resume(self, global_step):
        self.curr_steps = global_step
