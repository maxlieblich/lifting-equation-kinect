import numpy as np
import math
import data_model
from utilities import *
from handler import *

#ANGLES = [(limb[0], limb[1], limb2[1]) for limb in data_model.KIN_TREE for limb2 in data_model.KIN_TREE if limb[1] is limb2[0]]

AXES = {"kin": (0,0,1), "qual": (0,0,1)}

ANGLES = [('ShoulderRight', 'ElbowRight', 'WristRight'), 
          ('ShoulderLeft', 'ElbowLeft', 'WristLeft'),
          ('HipRight', 'KneeRight', 'AnkleRight'),
          ('HipLeft', 'KneeLeft', 'AnkleLeft')]

# def angle(Q1, axis=(0,0,1), degrees=True):
#     """computes the cosine of the angle between transformed axis and axis for
#     orthogonal transformation given by a quaternion.
#     For qualysis: take axis to be (0,0,1) for the joint angles.
#     For kinect: take axis to be (0,1,0) for the joint angles
#     Fed into handler: degreeify function for visibility"""
#     # probably unnecessary to normalize; maybe take this out
#     old_axis = V.Vector(axis)
#     new_axis = Q1.normalized().asRotation()(old_axis)#act(Q1, axis)
#     # sign change to compensate for what I assume is the fact that this gives the complement
#     # check in practice
#     return math.pi - new_axis.angle(old_axis)


def degreeify(A):
    return 180 / math.pi * np.arccos(A)
    
def basic_degreeify(A):
    return 180 / math.pi * np.array(A)    

class AngleHandlers(object):
    """Handle angles at all joints lying between two others.
    Totally general collection of handlers for each triplet of attached joints"""

    def __init__(self, DM):
        for kind in ['kin', 'qual']:
            setattr(self, kind, HandlerObject())
            for level in ['raw', 'smooth']:
                setattr(getattr(self, kind), level, dict())
                d = getattr(getattr(self, kind), level)
                for angle in ANGLES:
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



def z_angle(data=None, bone=None, method='quat'):
    if data is None:
        raise Exception("No data input")
    if method == 'quat':
        stuff = np.array(data[data_model.kin_bone_name(bone)])
        return stuff.T[0]*stuff.T[0] - stuff.T[1]*stuff.T[1] - stuff.T[2]*stuff.T[2] + stuff.T[3]*stuff.T[3]
    else: # kin is using savitzky-golay and model params are just residual moduli
        for angle in ANGLES:
            if angle[1] == bone[0]:
                first_joint = np.array(data[angle[0]])
                second_joint = np.array(data[angle[1]])
                third_joint = np.array(data[angle[2]])
                first_limb = first_joint - second_joint
                second_limb = third_joint - second_joint
                first_limb_normalized = normalize_rows(first_limb)
                second_limb_normalized = normalize_rows(second_limb)
                return np.clip(-1*np.sum((first_limb_normalized * second_limb_normalized), axis=-1), -1.0, 1.0)
#        if bone != ('HipCenter', 'HipCenter'):
#            raise Exception("No angle in ANGLES matches for computing this angle")
                
class QuatAngleHandlers(object):
    """Handle angles using quaternions. Slightly subtle: angles at shoulders
    should be chosen with more meaning"""

    def __init__(self, DM):
        for kind in ['kin', 'qual']:
            setattr(self, kind, HandlerObject())
            for level in ['raw', 'smooth']:
                setattr(getattr(self, kind), level, dict())
                d = getattr(getattr(self, kind), level)
                for bone in data_model.KIN_TREE:
                    if kind is 'kin' or bone in data_model.BONE_MAPPING:
                        if kind == 'kin' and DM.kin.smoothing == 'SG':# and bone != ('HipCenter', 'HipCenter'):
                            method = 'joint'
                        else:
                            method = 'quat'
                        handler = Handler(DM)
                       # stuff = getattr(getattr(DM, kind).data, level)[data_model.kin_bone_name(bone)]
                        handler.axis = AXES[kind]
                       # TEMP: FOLLOWING ASSUMES ACTING ON Z AXIS WITH NORMALIZED QUATERION
                        handler.raw_func = z_angle(data=getattr(getattr(DM, kind).data, level), bone=bone, method=method)
                        handler.times = getattr(DM, kind).data.raw['Time']
                        handler.time_shift = 0
                        handler.vis_func = degreeify#basic_degreeify
                        d[data_model.kin_bone_name(bone)] = handler