import numpy as np

import config


def P(t: float) -> float:
    if t < config.TIME_TO_START_ENGINE:
        return 0
    return config.PULL


def motion_system(*,
    P, alpha, tau, X, Y, mass, gravity_force, moment, moment_of_inertia, v_xz, v_yz, omega
) -> np.ndarray:
    d_v_xz = (P * np.cos(alpha + tau) - X * np.cos(tau) - Y * np.sin(tau)) / mass
    d_v_yz = (P * np.sin(alpha + tau) - X * np.sin(tau) + Y * np.cos(tau) - gravity_force) / mass
    d_omega = moment / moment_of_inertia
    d_x_z = v_xz
    d_y_z = v_yz
    d_tau = omega
    return (d_v_xz, d_v_yz, d_omega, d_x_z, d_y_z, d_tau)
