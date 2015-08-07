import numpy as np
from utilities import *
from handler import *

class HorizLocHandler(object):
    """Deal with horizontal location for NIOSH lifting equation"""
    def __init__(self, DM):
        for kind in ['kin', 'qual']:
            setattr(self, kind, HandlerObject())
            for level in ['raw', 'smooth']:
                handler = Handler(DM)
                handler.left_hand = getattr(getattr(DM, kind).data, level)['HandLeft']
                handler.right_hand = getattr(getattr(DM, kind).data, level)['HandRight']
                handler.left_foot = getattr(getattr(DM, kind).data, level)['FootLeft']
                handler.right_foot = getattr(getattr(DM, kind).data, level)['FootRight']
                handler.times = getattr(DM, kind).data.raw['Time']
                handler.raw_data = np.column_stack((handler.left_hand,
                                                 handler.right_hand,
                                                 handler.left_foot,
                                                 handler.right_foot))
                handler.raw_vec = 0.5 * (handler.left_hand[:, 0:2] + handler.right_hand[:, 0:2] -\
                                         handler.left_foot[:, 0:2] - handler.right_foot[:, 0:2])
                handler.raw_func = np.sqrt((handler.raw_vec)**2).sum(1)
                setattr(getattr(self, kind), level, handler)


class VertLocHandler(object):
    """Deal with vertical location for NIOSH lifting equation"""
    def __init__(self, DM):
        for kind in ['kin', 'qual']:
            setattr(self, kind, HandlerObject())
            for level in ['raw', 'smooth']:
                handler = Handler(DM)
                handler.left_hand = getattr(getattr(DM, kind).data, level)['HandLeft']
                handler.right_hand = getattr(getattr(DM, kind).data, level)['HandRight']
                handler.left_foot = getattr(getattr(DM, kind).data, level)['FootLeft']
                handler.right_foot = getattr(getattr(DM, kind).data, level)['FootRight']
                handler.times = getattr(DM, kind).data.raw['Time']
                handler.raw_data = np.column_stack((handler.left_hand,
                                                 handler.right_hand,
                                                 handler.left_foot,
                                                 handler.right_foot))
                handler.raw_func = 0.5 * (handler.left_hand[:, 2] + handler.right_hand[:, 2] -\
                                         handler.left_foot[:, 2] - handler.right_foot[:, 2])
                setattr(getattr(self, kind), level, handler)