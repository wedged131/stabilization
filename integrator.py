from collections.abc import Iterable, Callable

import numpy as np

import config


def integrator_RK4(
        init_params: Iterable,
        system,
        stopreason,
        t0: float = 0.0,
        dt: float = 1e-5,
        max_step_count: int = config.MAX_STEP_COUNT) -> np.ndarray:
    timelist = np.arange(t0, t0 + max_step_count * dt, dt)
    X = init_params.copy()
    res_len = len(X) + 1
    res = np.zeros((max_step_count, res_len))
    res[0, 0] = timelist[0]
    res[0, 1:res_len] = X
    stop_index = 0
    for i, t in enumerate(timelist):
        if i == 0: continue
        if stopreason(t, X):
            stop_index = i
            break
        k1 = system(t, X, dt)
        k2 = system(t + 0.5 * dt, X + k1 * 0.5 * dt, dt)
        k3 = system(t + 0.5 * dt, X + k2 * 0.5 * dt, dt)
        k4 = system(t + dt, X + k3 * dt, dt)
        X += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        res[i, 0] = t
        res[i, 1:res_len] = X
    else:
        stop_index = max_step_count
    return res[0:stop_index, :]


def integrate_step(value: float, derivate: float, step: float) -> float:
    return value + derivate * step
