import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent) + '\\')

import numpy as np

from integrator import integrator_RK4
from graphics import base_graph


def main():
    x = np.arange(0, 20, 0.5)
    y = x**2


if __name__ == "__main__":
    main()
