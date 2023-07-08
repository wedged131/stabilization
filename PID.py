import numpy as np

import config

def PID(input_signal: float) -> float:
    proportional_part = input_signal * config.PID_COEFFICIENTS["proportional"]
    integral_part = input_signal * config.PID_COEFFICIENTS["integral"]
    differential_part = input_signal * config.PID_COEFFICIENTS["differential"]

    output_signal = sum((proportional_part, integral_part, differential_part))
    return output_signal
