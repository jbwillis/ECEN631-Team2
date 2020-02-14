"""
Library of the various control tools we are using
for the autonomous car project
"""

import numpy as np

def decisionGridGaussian(nx, ny, sigx=1., sigy=1., gain=1.):
    """
    Decision grid based on a gaussian distribution function
    """
    x, y = np.meshgrid(np.linspace(-nx/2, nx/2, nx), np.linspace(0, ny, ny))

    dg = np.exp(-((x/(2.*sigx))**2 + (y/(2.*sigy))**2))

    m = np.amax(dg)
    dg = gain*dg/m

    # negate half of the image
    dg[..., int(nx/2):] *= -1

    return dg, x, y

def plt_decisionGrid(dg, x, y, block=True):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, dg)
    plt.show(block=block)

if __name__=="__main__":
    dg, x, y = decisionGridGaussian(20,20, sigx=3., sigy=5., gain=10.)
    plt_decisionGrid(dg, x, y)
