import numpy as np

import config

def PID(prev_value: float, next_value: float, dt: float) -> float:
    proportional_part = next_value * config.PID_COEFFICIENTS["proportional"]
    integral_part = (prev_value + next_value * dt) * config.PID_COEFFICIENTS["integral"]
    differential_part = (next_value - prev_value) / dt * config.PID_COEFFICIENTS["differential"]

    output_signal = sum((proportional_part, integral_part, differential_part))
    return output_signal
