import numpy as np
import matplotlib.pyplot as plt

def base_graph(x, y, *, show: bool = True, save: bool = False, savename: str | None = None):
    plt.figure()
    plt.plot(x, y)
    if show:
        plt.show()
    if save:
        plt.savefig(savename)