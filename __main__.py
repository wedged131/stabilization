import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent) + '/')

import numpy as np
import matplotlib.pyplot as plt

import config
from integrator import integrator_RK4
from motion_system import motion_system
from aerodynamic import aerodynamic
from actuator import actuator
from PID import PID
from graphics import base_graph


def P(t):
    if t < config.TIME_TO_START_ENGINE:
        return 0
    return config.PULL


def m(t):
    return config.MASS


def stopreason(t: float, vector: np.ndarray) -> bool:
    return t > config.TIME_TO_SIMULATE


def system(t: float, vector, dt) -> np.ndarray:
    res = np.zeros_like(vector)
    res[0], res[1], res[2], res[3], res[4], res[5], n_y_ = motion_system(
        P=vector[14], alpha=vector[8], tau=vector[5], X=vector[9], Y=vector[10], mass=vector[12],
        moment=vector[11], moment_of_inertia=vector[13], v_xz=vector[0], v_yz=vector[1], omega=vector[2], n_y=vector[6]
    )
    res[6] = n_y_ / dt
    res[7] =( PID(prev_value=-vector[2]) - vector[7] )/ dt
#    res[7] = actuator(delta_target, vector[7]) / dt
    res[8] = 0
    res[9], res[10], res[11] = aerodynamic(
        v_xz=vector[0], v_yz=vector[1], alpha=vector[8], delta=vector[7], H=vector[4]
    )
    res[9] /= dt
    res[10] /= dt
    res[11] /= dt
    res[12] = 0
    res[13] = 0
    res[14] = (P(t) - vector[14]) / dt
    return res


def main():
    dt = 0.001
    timelist = np.arange(0, 3, dt)
    res_len = 16 + 1
    res = np.zeros((len(timelist), res_len))

# start data
    _v_xz = 298                                     # 0
    _v_yz = -5                                      # 1
    _omega = np.radians(-10)                        # 2
    _x_z = 0                                        # 3
    _y_z = 3000                                     # 4
    _tau = np.radians(-5)                           # 5
    _n_y = 0                                        # 6
    _delta_target = 0                               # 7
    _delta_actual = 0                               # 8
    _alpha = 0                                      # 9
    _X = 0                                          # 10
    _Y = 0                                          # 11
    _M_aero = 0                                     # 12
    _mass = m(0)                                    # 13
    _moment_of_inertia = config.MOMENT_OF_INERTIA   # 14
    _pull = P(0)                                    # 15

# start simulation
    for i, t in enumerate(timelist):
        res[i, 0] = t
        res[i, 1:] = _v_xz, _v_yz, _omega, _x_z, _y_z, _tau, _n_y, _delta_target, _delta_actual, _alpha, _X, _Y, _M_aero, _mass, _moment_of_inertia, _pull
        _omega_prev = _omega
    # aerodynamics
        _X, _Y, _M_aero = aerodynamic(
            v_xz=_v_xz, v_yz=_v_yz, alpha=_alpha, delta=_delta_actual, H=_y_z
        )
        _pull = P(t)
        _mass = m(t)
    # motion_system
        motion_derivations = motion_system(
            P=_pull, alpha=_alpha, tau=_tau, X=_X, Y=_Y, mass=_mass, moment=_M_aero, moment_of_inertia=_moment_of_inertia,
            v_xz=_v_xz, v_yz=_v_yz, omega=_omega, n_y=_n_y
        )
        _v_xz += motion_derivations[0] * dt
        _v_yz += motion_derivations[1] * dt
        _omega += motion_derivations[2] * dt
        _x_z += motion_derivations[3] * dt
        _y_z += motion_derivations[4] * dt
        _tau += motion_derivations[5] * dt
        _n_y = motion_derivations[6]
    # PID
        _delta_target = PID(_omega_prev, _omega, dt)
    # actuator
        _delta_actual = actuator(_delta_target)

# start graphical vizualization
    t = res[:, 0]
    labels = [
        r"$v_{xz}$",
        r"$v_{yz}$",
        r"$\omega$",
        r"$x_z$",
        r"$y_z$",
        r"$\tau$",
        r"$n_y$",
        r"$\delta_{target}$",
        r"$\delta_{actual}$",
        r"$\alpha$",
        r"$X$",
        r"$Y$",
        r"$M_{aero}$",
        r"$mass$",
        r"$I$",
        r"$P$",
    ]
    for i in [2]:
        ylabel = labels[i]
        if i in [2, 5, 7, 8]:
            base_graph(t, np.degrees(res[:, i+1]), "$t, с$", ylabel, show=False)
        else:
            base_graph(t, res[:, i+1], "$t, с$", ylabel, show=False)
    # base_graph(res[:, 4], res[:, 5], "$x, м$", "y, м", show=False)
    plt.show()

if __name__ == "__main__":
    main()
