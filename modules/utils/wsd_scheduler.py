from math import sqrt


def warmup_stable_decay_lambda(total_steps, warmup_steps, cooldown_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif step > total_steps - cooldown_steps:
            return 1 - sqrt((step - (total_steps - cooldown_steps)) / cooldown_steps)
        else:
            return 1.0

    return lr_lambda
