import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools

# parsing data
data = np.genfromtxt(sys.argv[1], delimiter=',')
sampler, collision, distribution, numParticles, radius, numSamples = sys.argv[1].split('.')[
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
if dim == 2:
    columns = 1
    rows = 1
    size = 5

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
    for id, sample_set in enumerate(sample_list):
        ax[-1].scatter(sample_set[:, i], sample_set[:, i+1],
                       s=size, c='C'+str(id))


plt.savefig("samples/" + sys.argv[1].split('/')[1].split('.')[0] + ".png")
