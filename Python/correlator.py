from scipy import fftpack, signal
from utilities import *
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelmax
import numpy as np

# build this to always use right elbow angle over maximal data time
# to produce universal shift, try not recorrelating?

def correlate_funcs(f, g, a, b, plot=False):
    THRESHOLD = 0.05
    SMOOTHING = 0.05
    # not right yet when f and g have different domains,
    # so g could slot into a segment of f....
    # window function?
    sync_times = np.arange(a, b, 0.001)
    f_preraw = f(sync_times)
    g_preraw = g(sync_times)

    # do a rough alignment with argmax?
    f_rough = UnivariateSpline(sync_times, f_preraw, s=SMOOTHING)
    g_rough = UnivariateSpline(sync_times, g_preraw, s=SMOOTHING)
    f_maxes = argrelmax(f_rough(sync_times))[0]
    g_maxes = argrelmax(g_rough(sync_times))[0]
    for m in f_maxes:
        if f_rough(m * 0.001) - f_rough(a) > THRESHOLD:
            f_max = m * 0.001
            break
    for m in g_maxes:
        if g_rough(m * 0.001) - g_rough(a) > THRESHOLD:
            g_max = m * 0.001
            break
    
    # these are not a good sign; what will happen?
    if 'f_max' not in locals():
        f_max = 0
        
    if 'g_max' not in locals():
        g_max = 0    
            
    # now start over!
    a = 0
    b = b - max(f_max, g_max)
    sync_times = np.arange(a, b, 0.001)
    f_preraw = f(sync_times + f_max)
    g_preraw = g(sync_times + g_max)
    
    # filter out nan values; get to the bottom of this!
    mask = np.isnan(f_preraw) + np.isnan(g_preraw)    
    sync_times = sync_times[mask == 0]
    f_raw = f_preraw[mask == 0]
    g_raw = g_preraw[mask == 0]

    f_mean = sum(f_raw) / len(f_raw)
    g_mean = sum(g_raw) / len(g_raw)
    f_prestream = f_raw - f_mean
    g_prestream = g_raw - g_mean
    f_mag = np.amax(np.abs(f_prestream))
    g_mag = np.amax(np.abs(g_prestream))
    f_stream = f_prestream / f_mag
    g_stream = g_prestream / g_mag
    
    
    h = np.hamming(len(sync_times))
            
    f_fft = fftpack.fft(f_stream * h)
    g_fft = fftpack.fft(g_stream * h)

    f_conj = f_fft.conjugate()
    g_conj = g_fft.conjugate()

    f_comp_g = np.abs(fftpack.ifft(f_conj * g_fft))
    g_comp_f = np.abs(fftpack.ifft(f_fft * g_conj))

    if plot:
        pylab.figure()
        pylab.plot(np.linspace(0, 1000, len(f_comp_g)), f_comp_g)
        
    f_to_g = np.argmax(f_comp_g)
    g_to_f = np.argmax(g_comp_f)

    if (f_to_g < g_to_f):
        return f_to_g * 0.001 + g_max - f_max
    else:
        return g_to_f * -0.001 + g_max - f_max


class Correlator(object):
    # two handlers A and B
    def __init__(self, DM):
        return

    def set_shift(self, DM):
        return


class ZCorrelator(Correlator):
    """An abstract class for using z coordinates of joints to correlate.
    Assumed that it has been fed clean data extracted from qualysis."""
    def __init__(self, DM):
        self.DM = DM
        self.qual = None # set this in subclasses!
        self.kin = None # set this in subclasses!
    
    def build(self):
        qual_times = self.DM.qual.data.raw['Time']
        kin_times = self.DM.kin.data.raw['Time']
        qual_end = qual_times[-1]
        kin_end = kin_times[-1]
        self.end = min(qual_end, kin_end)
        qual_mask = (qual_times > self.end)
        kin_mask = (kin_times > self.end)
        kin_idx = np.argmax(kin_mask) if np.argmax(kin_mask) > 0 else len(kin_times)
        qual_idx = np.argmax(qual_mask) if np.argmax(qual_mask) > 0 else len(qual_times)
        self.kin_times = kin_times[0:kin_idx]
        self.qual_times = qual_times[0:qual_idx]
        self.kin = self.kin[0:kin_idx, :]
        self.qual = self.qual[0:qual_idx, :]
    
    def set_shift(self):
        a = 0
        b = self.end
        f = UnivariateSpline(self.kin_times, self.kin[:, 2], s=0)
        g = UnivariateSpline(self.qual_times, self.qual[:, 2], s=0)
        self.DM.time_shift = correlate_funcs(f, g, a, b)


class WristRightCorrelator(ZCorrelator):
    # this has to change with new qualysis data model
    # would have been better to use part of the mapping here....
    def __init__(self, DM):
        ZCorrelator.__init__(self, DM)
        self.qual = self.DM.qual.data.raw['Rw']
        self.kin = self.DM.kin.data.raw['WristRight']
        self.build()


class WristsCorrelator(ZCorrelator):
    """Take the sum of the z-coordinates of both wrists"""
    def __init__(self, DM):
        ZCorrelator.__init__(self, DM)
        # maybe better to use a DM mapping to avoid tying this part directly
        # to the representation of qualysis data....
        self.qual = self.DM.qual.data.raw['Rw'] + self.DM.qual.data.raw['Lw']
        self.kin = self.DM.kin.data.raw['WristRight'] + self.DM.kin.data.raw['WristLeft']
        self.build()
