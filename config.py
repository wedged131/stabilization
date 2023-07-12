from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent


MAX_STEP_COUNT = 1_000_000
CHARACTERISTIC_AREA = 0.09621
MOMENT_OF_INERTIA = 300.0
MASS = 350.0
CHARACTERISTIC_LENGTH = 3.85
PULL = 15000.0

TIME_TO_SIMULATE = 5.0
TIME_TO_START_ENGINE = 0.2

g = 9.80665
GRAVITY_FORCE = MASS * g

PID_COEFFICIENTS = {
    "proportional": 5,
    "integral": 5,
    "differential": 0.002
}

def read_table(filepath: Path) -> tuple[tuple[np.ndarray], np.ndarray]:
    input_table = pd.read_csv(filepath, sep=",")
    Mach = input_table["Mach"].unique()
    alpha = input_table["alpha"].unique()
    delta = input_table["delta"].unique()
    points = (Mach, alpha, delta)
    output_table = np.zeros((Mach.size, alpha.size, delta.size))
    for i, M in enumerate(Mach):
        for j, a in enumerate(alpha):
            for k, d in enumerate(delta):
                tmp = input_table[input_table["Mach"] == M]
                tmp = tmp[tmp["alpha"] == a]
                tmp = tmp[tmp["delta"] == d]
                output_table[i][j][k] = tmp['parameter'].values[0]
    return points, output_table

Cx_points, Cx_values = read_table(BASE_DIR.joinpath("Cx_table.csv"))
Cy_points, Cy_values = read_table(BASE_DIR.joinpath("Cy_table.csv"))
Mz_points, Mz_values = read_table(BASE_DIR.joinpath("Mz_table.csv"))
Cx_values *= 5
Cy_values *= 10
Mz_values *= 10
