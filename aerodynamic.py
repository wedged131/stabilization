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



def Cx(Mach, alpha, delta) -> float:
    return interpolate.interpn(
        config.Cx_points,
        config.Cx_values,
        (Mach, np.degrees(alpha), np.degrees(delta))
    )

def Cy(Mach, alpha, delta) -> float:
    return interpolate.interpn(
        config.Cy_points,
        config.Cy_values,
        (Mach, np.degrees(alpha), np.degrees(delta))
    )

def Mz(Mach, alpha, delta) -> float:
    return interpolate.interpn(
        config.Mz_points,
        config.Mz_values,
        (Mach, np.degrees(alpha), np.degrees(delta))
    )

def aerodynamic(*,
        Mach: float,
        alpha: float = 0,
        delta: float,
        H: float) -> np.ndarray:
    Cx_ = Cx(Mach, alpha, delta)
    Cy_ = Cy(Mach, alpha, delta)
    Mz_ = Mz(Mach, alpha, delta)
    dX = Cx_ * q(Mach, H) * config.CHARACTERISTIC_AREA
    dY = Cy_ * q(Mach, H) * config.CHARACTERISTIC_AREA
    dM = Mz_
    return (dX, dY, dM)
