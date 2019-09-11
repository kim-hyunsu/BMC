from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import kstest

from experiments import distributions as dist


def coloring(id, numParticles):
    ratio = id / (numParticles+1)
    return (
        255 * ratio,
        225 * (1 - ratio),
        255 * ((id + 1) % 2)
    )


data = np.genfromtxt(sys.argv[1], delimiter=',')
sampler, collision, distribution, numParticles, radius, numSamples = sys.argv[1].split('.')[
    0].split('_')
length = len(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ids = np.unique(data[:, 0])
id_list = data[:, 0]
for id in ids:
    samples = data[1000:][:, 5:]
    print("test stat", kstest(samples, dist.AsymMOG2d_cdf))
    x, y = np.extract(id_list == id, data[:, 5]), np.extract(
        id_list == id, data[:, 6])
    interval = [-10, 10]
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=80, range=[interval, interval])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(
        xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.1 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',
             color=coloring(id, numParticles))

# target function
X = np.arange(interval[0], interval[1], 0.125)
Y = np.arange(interval[0], interval[1], 0.125)
X, Y = np.meshgrid(X, Y)
xn, yn = X.shape
W1 = X*0
W2 = X*0
for xk in range(xn):
    for yk in range(yn):
        w1 = float(dist.AsymMOG2d(np.array([X[xk, yk], Y[xk, yk]]))) * 270
        W1[xk, yk] = w1
        w2 = float(dist.AsymMOG2d(np.array([X[xk, yk], Y[xk, yk]]))) * 800
        W2[xk, yk] = w2

# ax.plot_wireframe(X, Y, W1, rstride=5, cstride=5, colors='blue')
# ax.plot_wireframe(X, Y, W2, rstride=10, cstride=10, colors='yellow')

fig.savefig(sys.argv[2], bbox_inches='tight')
