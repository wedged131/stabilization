import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent) + '\\')

import numpy as np
import matplotlib.pyplot as plt

import config
from integrator import integrator_RK4
from motion_system import motion_system
from aerodynamic import aerodynamic
from actuator import actuator
from PID import PID
from graphics import base_graph


def P(t: float) -> float:
    if t < config.TIME_TO_START_ENGINE:
        return 0
    return config.PULL


def stopreason(t: float, vector: np.ndarray[float]) -> bool:
    return t > config.TIME_TO_SIMULATE


def system(t: float, vector: np.ndarray[float], dt) -> np.ndarray:
    res = np.zeros_like(vector)
    res[0], res[1], res[2], res[3], res[4], res[5], n_y_ = motion_system(
        P=vector[14], alpha=vector[8], tau=vector[5], X=vector[9], Y=vector[10], mass=vector[12],
        moment=vector[11], moment_of_inertia=vector[13], v_xz=vector[0], v_yz=vector[1], omega=vector[2], n_y=vector[6]
    )
    res[6] = n_y_ / dt
    delta_target = PID(input_signal=n_y_)
    res[7] = actuator(delta_target, vector[7]) / dt
    res[8] = 0
    res[9], res[10], res[11] = aerodynamic(
        v_xz=vector[0], v_yz=vector[1], alpha=vector[8], delta=vector[7], H=vector[4],
        X=vector[9], Y=vector[10], M_aero=vector[11]
    )
    res[9] /= dt
    res[10] /= dt
    res[11] /= dt
    res[12] = 0
    res[13] = 0
    res[14] = (P(t) - vector[14]) / dt
    return res


def main():
    init_conditions = [
        298,                        # ( 0) v_xz
        -5,                         # ( 1) v_yz
        np.radians(-10),            # ( 2) omega
        0,  # ----------------------# ( 3) x_z
        3000,                       # ( 4) y_z
        np.radians(-5),             # ( 5) tau
        0,  # ----------------------# ( 6) n_y
        0,  # ----------------------# ( 7) delta_actual
        np.radians(0.5),              # ( 8) alpha
        0,  # ----------------------# ( 9) X
        0,  # ----------------------# (10) Y
        0,  # ----------------------# (11) M_aero
        config.MASS,                # (12) mass
        config.MOMENT_OF_INERTIA,   # (13) moment_of_inertia
        P(0)                        # (14) pull
    ]
    res = integrator_RK4(
        init_conditions,
        system,
        stopreason,
        dt=0.001
    )
    t = res[:, 0]
    labels = [
        r"$v_{xz}$",
        r"$v_{yz}$",
        r"$\omega$",
        r"$x_z$",
        r"$y_z$",
        r"$\tau$",
        r"$n_y$",
        r"$\delta_{actual}$",
        r"$\alpha$",
        r"$X$",
        r"$Y$",
        r"$M_{aero}$",
        r"$mass$",
        r"$I$",
        r"$P$",
    ]
    for i in [2, 5, 6, 7, 11]:
        ylabel = labels[i]
        if i in [2, 5, 7, 8]:
            base_graph(t, np.degrees(res[:, i+1]), "$t, с$", ylabel, show=False)
        else:
            base_graph(t, res[:, i+1], "$t, с$", ylabel, show=False)
    base_graph(res[:, 4], res[:, 5], "$x, м$", "y, м", show=False)
    plt.show()

if __name__ == "__main__":
    main()
