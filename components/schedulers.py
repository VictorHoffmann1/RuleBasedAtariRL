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