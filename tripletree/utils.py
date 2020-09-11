import matplotlib
import numpy as np
from scipy.spatial import minkowski_distance

# Colour normaliser.
# From https://github.com/mwaskom/seaborn/issues/1309#issue-267483557
class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=None, clip=False):
        #if vmin == vmax: self.degenerate = True
        #else: self.degenerate = False
        if midpoint == None: self.midpoint = (vmax + vmin) / 2
        else: self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        #if self.degenerate: return 'w'
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# A custom colour map.
# cdict = {'red':   [[0.0,  1.0, 1.0],
#                    #[0.5,  0.25, 0.25],
#                    [0.5, 0.8, 0.8],
#                    [1.0,  0.0, 0.0]],
#          'green': [[0.0,  0.0, 0.0],
#                    #[0.5,  0.25, 0.25],
#                    [0.5, 0.6, 0.6],
#                    [1.0,  0.8, 0.8]],
#          'blue':  [[0.0,  0.0, 0.0],
#                    #[0.5,  1.0, 1.0],
#                    [0.5, 0.0, 0.0],
#                    [1.0,  0.0, 0.0]]}  
mid = 160/256  
cdict = {'red':   [[0.0, 245/256, 245/256],
                   #[0.5, mid, mid],
                   [0.5, 143/256, 143/256],
                   [1.0, 41/256, 41/256]],
         'green': [[0.0, 64/256, 64/256],
                   #[0.5, mid, mid],
                   [0.5, 64/256, 64/256],
                   [1.0, 64/256, 64/256]],
         'blue':  [[0.0, 41/256, 41/256],
                   #[0.5, mid, mid],
                   [0.5, 135.5/256, 135.5/256],
                   [1.0, 230/256, 230/256]]}                 
custom_cmap = matplotlib.colors.LinearSegmentedColormap('custom_cmap', segmentdata=cdict)

# Fast way to calculate running mean.
# From https://stackoverflow.com/a/43200476
import scipy.ndimage.filters as ndif
def running_mean(x, N):
    x = np.pad(x, N // 2, mode='constant', constant_values=(x[0],x[-1]))
    return ndif.uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]

# Convert sequence of bits (e.g. leaf address) to an integer.
# NOTE: Need to prepend a "1" to disambiguate between 00000 and 000.
def bits_to_int(bits):
    out = 0
    for bit in [1] + list(bits): out = (out << 1) | bit
    return out

# Convert the other way.
def int_to_bits(integer): return tuple(int(i) for i in bin(integer)[3:])

# Frechet distance computation with scaled dimensions.
# Slightly changed from https://github.com/cjekel/similarity_measures/blob/master/similaritymeasures/similaritymeasures.py.
def scaled_frechet_dist(X, Y, scales, p):    
    X = X * scales
    Y = Y * scales
    n, m = len(X), len(Y)
    ca = np.multiply(np.ones((n, m)), -1)
    ca[0, 0] = minkowski_distance(X[0], Y[0], p=p)
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], minkowski_distance(X[i], Y[0], p=p))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], minkowski_distance(X[0], Y[j], p=p))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                           minkowski_distance(X[i], Y[j], p=p))
    return ca[n-1, m-1]