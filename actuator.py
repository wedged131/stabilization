import numpy as np


def actuator(input_signal: float, current_value: float) -> float:
    output_signal = input_signal - current_value
    return output_signal
