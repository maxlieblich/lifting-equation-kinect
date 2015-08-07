import numpy as np
from scipy.interpolate import UnivariateSpline
import pylab
import matplotlib
import math
import data_model
from utilities import *
from handler import *

#ANGLES = [(limb[0], limb[1], limb2[1]) for limb in data_model.KIN_TREE for limb2 in data_model.KIN_TREE if limb[1] is limb2[0]]

def degreeify(A):
    return 180 / math.pi * np.arccos(A)

class QuatAngleHandlers(object):
    """Handle angles at all joints lying between two others.
    Totally general collection of handlers for each triplet of attached joints"""

    def __init__(self, DM):
        for kind in ['kin', 'qual']:
            setattr(self, kind, HandlerObject())
            for level in ['raw', 'smooth']:
                setattr(getattr(self, kind), level, dict())
                d = getattr(getattr(self, kind), level)
                for angle in data_model.ANGLES:
                    handler = Handler(DM)
                    handler.first_joint = getattr(getattr(DM, kind).data, level)[angle[0]]
                    handler.second_joint = getattr(getattr(DM, kind).data, level)[angle[1]]
                    handler.third_joint = getattr(getattr(DM, kind).data, level)[angle[2]]
                    handler.first_limb = handler.first_joint - handler.second_joint
                    handler.second_limb = handler.third_joint - handler.second_joint
                    handler.first_limb_normalized = normalize_rows(handler.first_limb)
                    handler.second_limb_normalized = normalize_rows(handler.second_limb)
                    handler.raw_func = np.sum((handler.first_limb_normalized * handler.second_limb_normalized), axis=-1)
                    handler.times = getattr(DM, kind).data.raw['Time']
                    handler.data = np.column_stack((handler.first_joint,
                                                    handler.second_joint,
                                                    handler.third_joint))
                    handler.time_shift = 0
                    handler.vis_func = degreeify
                    d[angle[1]] = handler