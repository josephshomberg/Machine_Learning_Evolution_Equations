FILENAME = 'data/128x128_DBC_kappa=4.7_batch_000.npz'
#FILENAME = 'test-dataset_128x128_DBC_kappa=4.7_count=10000.npz'
NUMBER = 1

import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import pylab as pylab

def load_real_samples(filename):
    # load compressed arrays
    data = load(filename, mmap_mode='r')
    # unpack arrays using correct keys
    X1, X2 = data['src'], data['tar']
    # sanity checks
    assert not np.any(np.isnan(X1))
    assert not np.any(np.isnan(X2))
    return [X1, X2]

a = load_real_samples(FILENAME)
print('Loaded shapes:', a[0].shape, a[1].shape)
print('One pair shape:', a[0][NUMBER].shape, a[1][NUMBER].shape)
print('max/min a[0]:', np.max(a[0]), np.min(a[0]))
print('max/min a[1]:', np.max(a[1]), np.min(a[1]))
print('means:', np.mean(a[0]), np.mean(a[1]))
print('variances:', np.var(a[0]), np.var(a[1]))

# plot src[NUMBER]
pylab.clf()
plt.axis('off')
plt.imshow(a[0][NUMBER].squeeze(), cmap='Greys', interpolation='none')
plt.savefig('phiNE.png', bbox_inches='tight', pad_inches=0)

# plot tar[NUMBER]
pylab.clf()
plt.axis('off')
plt.imshow(a[1][NUMBER].squeeze(), cmap='Greys', interpolation='none')
plt.savefig('phiIC.png', bbox_inches='tight', pad_inches=0)
