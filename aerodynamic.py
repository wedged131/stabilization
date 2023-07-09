from pathlib import Path

import numpy as np
import scipy.interpolate as interpolate

import config
from atmosphere import atmo


def Mach_to_v(Mach: float, H: float = 0) -> float:
    '''Конвертирует скорость в Махах в скорость в м/с, учитывая высоту (стандартное значение высоты = 0 м)'''
    return Mach * atmo.a(H)

def v_to_Mach(v: float, H: float = 0) -> float:
    '''Конвертирует скорость в м/с в скорость в Махах, учитывая высоту (стандартное значение высоты = 0 м)'''
    return v / atmo.a(H)

def q(Mach: float, H: float = 0) -> float:
    '''
    Расчет скоростного напора
    [Литература: А. А. Лебедев и Л.С. Чернобровкин - "Динамика полета" (стр. 33)]

    Ввод:   H: float - высота полета [м]
            M: float - число Маха [-]
    Вывод:  q: float - скоростной напор [кг/(м*с^2)]
    '''
    v = Mach_to_v(Mach, H)
    q = atmo.rho(H) * v * v / 2
    return q


def clamped_delta(delta: float):
    if delta > 10:
        return 10
    if delta < -10:
        return -10
    return delta


def Cx(Mach, alpha, delta) -> float:
    # print(Mach, np.degrees(alpha), np.degrees(delta))
    _delta = clamped_delta(np.degrees(delta))
    return interpolate.interpn(
        config.Cx_points,
        config.Cx_values,
        (Mach, np.degrees(alpha), _delta)
    )[0]

def Cy(Mach, alpha, delta) -> float:
    _delta = clamped_delta(np.degrees(delta))
    return interpolate.interpn(
        config.Cy_points,
        config.Cy_values,
        (Mach, np.degrees(alpha), _delta)
    )[0]

def Mz(Mach, alpha, delta) -> float:
    _delta = clamped_delta(np.degrees(delta))
    return interpolate.interpn(
        config.Mz_points,
        config.Mz_values,
        (Mach, np.degrees(alpha), _delta)
    )[0]

def _Mach(v_xz: float, v_yz: float, H: float) -> float:
    v = np.sqrt( v_xz * v_xz + v_yz * v_yz )
    return v_to_Mach(v, H)

def aerodynamic(*,
        v_xz: float,
        v_yz: float,
        alpha: float = 0,
        delta: float,
        H: float,
        X: float,
        Y: float,
        M_aero: float) -> tuple:
    Mach = _Mach(v_xz, v_yz, H)
    Cx_ = Cx(Mach, alpha, delta)
    Cy_ = Cy(Mach, alpha, delta)
    Mz_ = Mz(Mach, alpha, delta)
    # print(Cx_, Cy_, Mz_)
    dX = Cx_ * q(Mach, H) * config.CHARACTERISTIC_AREA - X
    dY = Cy_ * q(Mach, H) * config.CHARACTERISTIC_AREA - Y
    dM = Mz_ * q(Mach, H) * config.CHARACTERISTIC_AREA * config.CHARACTERISTIC_LENGTH - M_aero
    return (dX, dY, dM)

# скорее всего тут надо будет добавлять все тоже самое, что и с перегрузкой,
# то есть минусовать старое значение и делить на dt