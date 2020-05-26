from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import kstest

from experiments import distributions as dist


data = np.genfromtxt(sys.argv[1], delimiter=',')
sampler, collision, distribution, numParticles, radius, numSamples = sys.argv[1].split('.csv')[
    0].split('_')
length = len(data)
id_list = data[length//10:, 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# x, y = np.extract(id_list > 1, data[length//10:][:, 5]
#                   ), np.extract(id_list > 1, data[length//10:][:, 6])
x, y = data[length//10:][:, 5], data[length//10:][:, 6]
minimum, maximum = min(np.amin(x), np.amin(y)), max(np.amax(x), np.amax(y))
interval = [-30, 30]
hist, xedges, yedges = np.histogram2d(
    x, y, bins=100, range=[interval, interval])

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
         color='C1')

fig.savefig("histogram/" + sys.argv[1].split('/')
            [1].split('.csv')[0] + ".png", bbox_inches='tight')
