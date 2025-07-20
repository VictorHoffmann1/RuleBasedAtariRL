import numpy as np


# Linear LR schedule
def linear_scheduler(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


# Exponential LR schedule
def exponential_scheduler(initial_value: float, final_value: float):
    log_initial = np.log(initial_value)
    log_final = np.log(final_value)

    def func(progress_remaining: float) -> float:
        # Convert progress_remaining (1 → 0) into fraction of progress (0 → 1)
        frac = 1.0 - progress_remaining
        log_lr = log_initial + frac * (log_final - log_initial)
        return float(np.exp(log_lr))

    return func

def get_lr(scheduler, lr, n_steps, final_lr=1e-5, total_steps=1e7):
    if scheduler == "exponential":
        final_step = lr * 10 ** ((np.log10(final_lr) - np.log10(lr)) / total_steps * n_steps)
        return exponential_scheduler(lr, final_step)
    elif scheduler == "linear":
        final_step = (final_lr - lr) / total_steps * n_steps + lr
        return linear_scheduler(lr, final_step)
    elif scheduler == "constant":
        return lr
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}. Use 'exponential', 'linear', or 'constant'.")