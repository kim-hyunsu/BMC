import matplotlib.pyplot as plt
import numpy as np
import sys

# parsing data
data = np.genfromtxt(sys.argv[1], delimiter=',')
sampler, collision, distribution, numParticles, radius, numSamples = sys.argv[1].split('.')[
    0].split('_')
length = len(data)

# get all ids
ids = np.unique(data[:, 0])
id_list = data[:, 0]

# calculate moment and classify it for each ids
all_moments = [[] for _ in range(int(numParticles[1:]))]
for id in ids:
    id = int(id)
    samples = np.array([np.extract(id_list == id, data[:, i])
                        for i in range(5, len(data[0]))])
    samples = samples.T
    moments = np.array([np.dot(samples[:i+1, 0], samples[:i+1, 1])/(i+1)
                        for i in range(len(samples))])
    print(moments)
    all_moments[id] = moments

# plot moment for each ids
for i, moments in enumerate(all_moments):
    plt.plot(np.arange(len(moments)), moments, color='C'+str(i))

plt.savefig("moment/" + sys.argv[1].split('/')[1].split('.')[0] + ".png")
