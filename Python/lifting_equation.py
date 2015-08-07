import correctors
from utilities import normalize_rows
from sklearn.externals import joblib
import numpy as np
from numpy.fft import rfft, fftfreq, fft, ifft, irfft
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelmin, argrelmax
from data_model import KIN_LABELS as KL
from position_pool import normalize_data
from body_model import BodyModel
import sav_gol

HL = correctors.lookup("Horizontal Location").func
VL = correctors.lookup("Vertical Location").func
AA = correctors.lookup("Asymmetry Angle").func

HLm = joblib.load("DataFiles/models/Horizontal Location").predict
VLm = joblib.load("DataFiles/models/Vertical Location").predict
AAm = joblib.load("DataFiles/models/Asymmetry Angle").predict

# ALL METRIC FOR NOW

LC = 23 # kg

FREQUENCIES = np.array([0.2, 0.5, 1, 2, 3, 4,
                        5, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, 15.5]) # 15.5 stands for >15

# duration-linked 0s: 4 from end, 6 from end, 8 from end
DURATIONS = [1, 2, 8] #means: (0,1] hr, (1,2] hrs, (2,8] hrs

CUTOFFS = {1: 14, 2: 12, 8: 10}

# these are the columns for V >= 30
FM_BASE_COLS = {1: [1.00, 0.97, 0.94, 0.91, 0.88, 0.84,
                    0.80, 0.75, 0.70, 0.60, 0.52, 0.45,
                    0.41, 0.37, 0.34, 0.31, 0.28, 0.00],
                2: [0.95, 0.92, 0.88, 0.84, 0.79, 0.72,
                    0.60, 0.50, 0.42, 0.35, 0.30, 0.26,
                    0.23, 0.21, 0.00, 0.00, 0.00, 0.00],
                8: [0.85, 0.81, 0.75, 0.65, 0.55, 0.45,
                    0.35, 0.27, 0.22, 0.18, 0.15, 0.13,
                    0.00, 0.00, 0.00, 0.00, 0.00, 0.00]}

COUPLINGS = {"Good": [1.0, 1.0], "Fair": [0.95, 1.00], "Poor": [0.9, 0.9]}


def duration_multiplier(duration, i, V):
    if V >= 30 * 2.54:
        return 1
    else:
        return i <= CUTOFFS[duration]


class LiftingEquation(object):
    def __init__(self, data=None, modeled=True, length_fit=False, floor=-2):
        self.floor = floor
        self.V = None # VerticalLocation at upper limit of reach
        self.H = None
        self.D = None # vertical distance traveled
        self.A = None
        self.F = None # lifts per minute
        self.origins = None # lift start times
        self.ends = None # lift end times
        self.BM = None # body model: fit to lengths
        self.data = None
        self.smoothed_VL = None
        self.times = None
        self.newtimes = None
        if not data is None:
            self.ingest_data(data, modeled=modeled, length_fit=length_fit)

    def ingest_data(self, data, modeled=True, length_fit=False):
        self.BM = BodyModel(data)
        if length_fit:
            self.BM.fit()
        self.data = data
        if modeled:
            n = normalize_data({'kin': self.BM.data})['kin']
            X = np.column_stack((n[key] for key in KL))
            self.model_data = normalize_rows(X)

        times = self.data['Time']
        newtimes = np.linspace(times[0], times[-1], 1000)

        self.times = times
        self.newtimes = newtimes

        floor = self.floor if self.floor else 0.5 * (np.mean(self.data['FootRight'].T[2]) + np.mean(self.data['FootLeft'].T[2]))
        #if (floor == None): floor = 0.5 * (np.mean(self.data['FootRight'].T[2]) + np.mean(self.data['FootLeft'].T[2]))
        vl = VL(self.BM.data, floor=floor)
        modeled_vl = vl + VLm(self.model_data) if modeled else vl
        subsampled_vl = UnivariateSpline(times, modeled_vl, s=0)(newtimes)
        self.smoothed_VL = UnivariateSpline(newtimes, sav_gol.savgol_filter(subsampled_vl, 31, 3), s=0) # smooth pretty hard to stop wobbling, overest?

        hl = HL(self.BM.data)
        modeled_hl = hl + HLm(self.model_data) if modeled else hl
        subsampled_hl = UnivariateSpline(times, modeled_hl, s=0)(newtimes)
        self.smoothed_HL = UnivariateSpline(newtimes, sav_gol.savgol_filter(subsampled_hl, 31, 3), s=0) # smooth pretty hard to stop wobbling, overest?

        aa = AA(self.BM.data)
        modeled_aa = aa + AAm(self.model_data) if modeled else aa
        subsampled_aa = UnivariateSpline(times, modeled_aa, s=0)(newtimes)
        self.smoothed_AA = UnivariateSpline(newtimes, sav_gol.savgol_filter(subsampled_aa, 31, 3), s=0) # smooth pretty hard to stop wobbling, overest?

        self.find_F()
        # self.find_origins()
        # self.find_ends()
        self.find_V()
        self.find_H()
        self.find_D()
        self.find_A()

    def find_F(self):
        # currently a kludge: do some simple fourier stuff
        # coming up: HMM or something like that to recognize the lifting motions
        # then count the in the period of interest
        # and use the lift characterization (maybe plus some local max/min) to find starts/ends
        times = self.data['Time']
        subtimes = np.linspace(times[0], times[-1], 1000)
        Zs = np.array(self.data['HandRight']).T[2] + np.array(self.data['HandLeft']).T[2]
        subZ = UnivariateSpline(times, Zs, s=0.5)(subtimes)
        Fs = subZ - np.mean(subZ)
        FF = rfft(Fs)
        mgft = np.abs(FF)
        i = np.argmax(mgft)
        self.F = fftfreq(len(Fs), (times[-1] - times[0]) / 60000.0)[i] # cycles per minute for dominant component
        filtered = irfft([FF[i] if i <= self.F else 0 for i in range(len(FF))])
        self.filtered = filtered
        self.origins = subtimes[argrelmin(filtered)]  # will need to pay attention to accidental near peaks
        self.ends = subtimes[argrelmax(filtered)]
        self.refine_origins()
        self.refine_ends()

    def refine_origins(self):
        if (self.origins is None): self.find_F()
        vls = self.smoothed_VL(self.times)
        raw_mins = self.times[argrelmin(vls)]
        self.origins = raw_mins[np.array([np.argmin(np.abs(raw_mins - orig)) for orig in self.origins])]

    def refine_ends(self):
        if (self.origins is None): self.find_F()
        vls = self.smoothed_VL(self.times)
        raw_maxs = self.times[argrelmax(vls)]
        self.ends = raw_maxs[np.array([np.argmin(np.abs(raw_maxs - en)) for en in self.ends])]
    
    def lows(self):
        vls = self.smoothed_VL(self.times)
        raw_mins = self.times[argrelmin(vls)]
        return raw_mins[np.array([np.argmin(np.abs(raw_mins - orig)) for orig in self.origins])]
    
    def highs(self):
        vls = self.smoothed_VL(self.times)
        raw_maxs = self.times[argrelmax(vls)]
        return raw_maxs[np.array([np.argmin(np.abs(raw_maxs - en)) for en in self.ends])]

    def find_V(self):
        self.V = np.mean(self.smoothed_VL(self.origins))

    def find_D(self):
        self.D = np.mean(self.smoothed_VL(self.ends)) - np.mean(self.smoothed_VL(self.origins))

    def find_H(self):
        self.H = np.mean(self.smoothed_HL(self.origins))

    def find_A(self):
        self.A = np.mean(self.smoothed_AA(self.origins))

    def HM(self):
        return 25.0 / max(25.0, 100 * self.H)

    def VM(self):
        return 1 - (0.003 * np.abs(max(0, min(175, 100 * self.V)) - 75))

    def DM(self):
        return 0.82 + 4.5 / min(max(25.0, 100 * self.D), 175)

    def AM(self):
        return 1 - 0.0032 * self.A if self.A < 135 else 0

    def FM(self, duration): # frequency multiplier, ave lifts per min over 15 mins
    # read about durations, etc.
        i = np.argmin(np.abs(self.F - FREQUENCIES))
        mult = duration_multiplier(duration, i, self.V)
        return FM_BASE_COLS[duration][i] * mult

    def CM(self, coupling): # coupling multiplier
        if self.V >= 30 * 2.54:
            return COUPLINGS[coupling][1]
        else:
            return COUPLINGS[coupling][0]

    def RWL(self, duration, coupling):
        return LC * self.HM() * self.VM() * self.DM() * \
               self.AM() * self.FM(duration) * self.CM(coupling)

    def print_all(self, duration=2, coupling='Fair'):
        print "V at origin: {0}".format(self.V)
        print "H: {0}".format(self.H)
        print "D: {0}".format(self.D)
        print "A: {0}".format(self.A)
        print "F: {0}".format(self.F)
        print "RWL: {0}".format(self.RWL(duration=duration, coupling=coupling))