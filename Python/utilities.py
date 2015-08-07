import numpy as np
import matplotlib as mpl
import pylab

# feed a numpy array, will normalize row vectors and output as a col

def normalize_rows(a):
    try:
        return a / np.sqrt(np.sum(a ** 2, axis=-1))[:, np.newaxis]#np.apply_along_axis(np.linalg.norm, -1, a)#
    except IndexError:
        return a / np.sqrt(np.sum(a ** 2, axis=-1))
#apply_along_axis(np.linalg.norm, 0, a)

def norm(a):
    return np.sqrt((a**2).sum(1))[:, np.newaxis]

# given a function and a range, return a new function that 
# outputs 0 where it is undefined in that range
def zero_undef(f, (a, b)):
    def h(t):
        # stupid normalization. more intelligent choice?
        if t < a or t > b: 
            return f(t)
        else:
            return 0
    return h

def quick_clean_mask(data):
    """Clean rows have 0"""
    return (data == 0).sum(1)

def quick_clean(data):
    """'Cleans' data by eliminating rows containing 0. OK for
    qualysis since 0 is the floor, but dangerous for Kinect. OTOH, 
    Kinect tries to interpolate, so probably don't need to clean it, 
    just possibly smooth it"""
    rows_missing_data = (data == 0).sum(1)
    return data[rows_missing_data == 0, :]
