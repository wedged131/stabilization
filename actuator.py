import numpy as np


def clamped_delta(delta: float):
    if delta > np.radians(10):
        return np.radians(10)
    if delta < np.radians(-10):
        return np.radians(-10)
    return delta


def actuator(input_signal: float) -> float:
    output_signal = clamped_delta(input_signal)
    return output_signal
