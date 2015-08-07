import numpy as np
from scipy.interpolate import UnivariateSpline
import pylab
import matplotlib
import math
from utilities import *
from handler import *
import data_model

class SegmentHandlers(object):
    """Collect and handle segment lengths"""
    def __init__(self, DM):
        for kind in ['kin', 'qual']:
            setattr(self, kind, HandlerObject())
            for level in ['raw', 'smooth']:
                setattr(getattr(self, kind), level, dict())
                d = getattr(getattr(self, kind), level)
                for seg in data_model.KIN_SEGMENTS:
                    handler = Handler(DM)
                    handler.first_joint = getattr(getattr(DM, kind).data, level)[seg[0]]
                    handler.second_joint = getattr(getattr(DM, kind).data, level)[seg[1]]
                    handler.limb = handler.first_joint - handler.second_joint
                    handler.raw_func = np.sqrt(np.sum((handler.limb * handler.limb), axis=-1))
                    handler.times = getattr(DM, kind).data.raw['Time']
                    handler.time_shift = 0
                    d[seg] = handler