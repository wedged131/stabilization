import numpy as np

import config


def motion_system(*,
    P, alpha, tau, X, Y, mass, moment, moment_of_inertia, v_xz, v_yz, omega, n_y
) -> np.ndarray:
    d_v_xz = (P * np.cos(alpha + tau) - X * np.cos(tau) - Y * np.sin(tau)) / mass
    d_v_yz = (P * np.sin(alpha + tau) - X * np.sin(tau) + Y * np.cos(tau) - mass * config.g) / mass
    d_omega = moment / moment_of_inertia
    d_x_z = v_xz
    d_y_z = v_yz
    d_tau = omega
    n_y_ = (Y + P * np.sin(alpha)) / (mass * config.g) - np.cos(tau)
    return (d_v_xz, d_v_yz, d_omega, d_x_z, d_y_z, d_tau, n_y_)
