import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import ordinal

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
    all_moments[id] = moments

# plot moment for each ids
for i, moments in enumerate(all_moments):
    plt.plot(np.arange(len(moments)), moments,
             color='C'+str(i), label=f'{ordinal(i)} particle')

# calculate moment for all ids
all_moments = np.array(all_moments)
moment = np.mean(all_moments, axis=0)

# plot moment for all ids
plt.plot(np.arange(len(moment)), moment, '--',
         color='black', label='Mean')

# plot ground truth
plt.plot(np.arange(len(moment)),
         [-3. for _ in range(len(moment))], '--', color='gray', label='Ground truth')

plt.xlabel('Iterations')
plt.ylabel('E[X1X2]')
plt.legend(loc='upper right', prop={'size': 8})
plt.savefig("moment/" + sys.argv[1].split('/')[1].split('.')[0] + ".png")
