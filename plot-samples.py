import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools

# parsing data
data = np.genfromtxt(sys.argv[1], delimiter=',')
sampler, collision, distribution, numParticles, radius, numSamples = sys.argv[1].split('.csv')[
    0].split('_')
# constants
dim = int(''.join(list(next(iter(())) if not i.isdigit()
                       else i for i in distribution[-2::-1]))[::-1])
particles = int(numParticles[1:])
radius = float(radius[1:])
samples = int(numSamples[1:])
data = data[samples//10:]

# get all ids
ids = np.unique(data[:, 0])
id_list = data[:, 0]

# image setting
fig = plt.figure(figsize=(36, 20))
columns = 9
rows = 5
size = 0.1
if dim <= 2:
    fig = plt.figure(figsize=(36, 15*(particles+1)))
    columns = 1
    rows = particles + 1
    size = 2.5

# split each id
sample_list = []
for id in ids:
    id = int(id)
    sample_set = np.array([np.extract(id_list == id, data[:, i])
                           for i in range(5, len(data[0]))])
    sample_list.append(sample_set.T)


# plot scatters
ax = []
for order, (i, j) in enumerate(itertools.combinations([n for n in range(dim)], 2)):
    ax.append(fig.add_subplot(rows, columns, order+1))
    ax[-1].set_title(f"Dim {i+1} vs. {j+1}")
    xlim = (min(np.amin(s[:, i]) for s in sample_list),
            max(np.amax(s[:, i]) for s in sample_list))
    ylim = (min(np.amin(s[:, j]) for s in sample_list),
            max(np.amax(s[:, j]) for s in sample_list))
    ax[-1].set_xlim(xlim)
    ax[-1].set_ylim(ylim)
    for id, sample_set in enumerate(sample_list):
        ax[-1].scatter(sample_set[:, i], sample_set[:, j],
                       s=size, c='C'+str(id))
    if dim <= 2:
        for id, sample_set in enumerate(sample_list):
            ax.append(fig.add_subplot(rows, columns, id+2))
            ax[-1].set_title(f"Particle {id+1}")
            ax[-1].set_xlim(xlim)
            ax[-1].set_ylim(ylim)
            ax[-1].scatter(sample_set[:, i], sample_set[:, j],
                           s=size, c='C'+str(id))


plt.savefig("samples/" + sys.argv[1].split('/')[1].split('.csv')[0] + ".png")
