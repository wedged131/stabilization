import numpy as np

from .graphics import base_graph

def main():
    x = np.arange(0, 20, 0.5)
    y = x**2
    base_graph(x, y)


if __name__ == "__main__":
    main()