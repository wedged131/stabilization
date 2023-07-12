import matplotlib.pyplot as plt

def base_graph(x, y, xlabel, ylabel, *, show: bool = True, save: bool = False, savename = None):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.plot(x, y)
    if show:
        plt.show()
    if save:
        plt.savefig(savename)
