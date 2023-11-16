class NoneScheduler:
    def __init__(self):
        pass

    def __call__(self):
        return 1

    def step(self):
        pass

    def resume(self, global_step):
        pass
