import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent) + '\\')

import numpy as np

import config
from integrator import integrator_RK4
from motion_system import motion_system
from aerodynamic import aerodynamic
from actuator import actuator
from PID import PID
from graphics import base_graph


def stopreason(t: float, vector: np.ndarray[float]) -> bool:
    return t > config.TIME_TO_SIMULATE


def system(t: float, vector: np.ndarray[float]) -> np.ndarray:
    res = np.zeros_like(vector)
    res[:] = motion_system(*vector[:])
    res[:] = aerodynamic(*vector[:])
    res[:] = actuator(*vector[:])
    res[:] = PID(*vector[:])
    res[:] = np.zeros()
    return res


def main():
    init_conditions = []
    res = integrator_RK4(
        init_conditions,
        system,
        stopreason,
        dt=0.001
    )
    t = res[:, 0]
    param_1 = res[:, 1]
    base_graph(t, param_1)


if __name__ == "__main__":
    main()
