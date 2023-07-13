import numpy as np

import config

def PID(base_value: float, input_value: float, dt: float) -> float:
    stored_value = (base_value - input_value)
    proportional_part = stored_value * config.PID_COEFFICIENTS["proportional"]
    integral_part = (PID.stored_value + stored_value * dt) * config.PID_COEFFICIENTS["integral"]
    differential_part = (PID.stored_value - stored_value) / dt * config.PID_COEFFICIENTS["differential"]

    output_signal = sum((proportional_part, integral_part, differential_part))
    PID.stored_value = stored_value
    return output_signal
