# AS OF 6/6/2013, NO MORE QUATERNIONS IN KINECT DATA: THEY ARE JUST THE 
# CANONICAL ONES. PERHAPS THE NEXT VERSION WILL HAVE MORE USEFUL QUATS
# AND WE CAN UNCOMMENT THE CODE

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import signal
import json
import pylab
import quaternions
import sav_gol
from utilities import normalize_rows

TIME_SCALE = 1000.0
HEADER = 7

# Butterworth params
RATE = 60
ORDER = 4
FREQ_CUTOFF = 7.0 / (RATE / 2.0)

QUAL_LABELS = ['Lw', 'Rw', 'Lsh', 'Rsh',
               'Leb', 'Reb', 'Lhip', 'Rhip',
               'Lknee', 'Rknee', 'Lank', 'Rank']

BASIC_QUAL_LABELS = ['Lw', 'Rw', 'Lsh', 'Rsh',
               'Leb', 'Reb', 'Lhip', 'Rhip',
               'Lknee', 'Rknee', 'Lank', 'Rank']

KIN_LABELS = ['HipCenter', 'Spine', 'ShoulderCenter', 'Head',
              'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
              'ShoulderRight', 'ElbowRight', 'WristRight',
              'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft',
              'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight',
              'FootRight']

# concatenation of kinlabels and quallabels
ALL_LABELS = ['HipCenter', 'Spine', 'ShoulderCenter', 'Head',
              'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
              'ShoulderRight', 'ElbowRight', 'WristRight',
              'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft',
              'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight',
              'FootRight', 'Lw', 'Rw', 'Lsh', 'Rsh',
              'Leb', 'Reb', 'Lhip', 'Rhip',
              'Lknee', 'Rknee', 'Lank', 'Rank']


KIN_UPPER_BODY = ('ShoulderRight', 'ElbowRight', 
                  'WristRight', 'ShoulderLeft', 
                  'ElbowLeft', 'WristLeft')

KIN_UPPER_BODY_EDGES = (('ShoulderRight', 'ElbowRight'), 
             ('ElbowRight', 'WristRight'), 
             ('ShoulderLeft', 'ElbowLeft'), 
             ('ElbowLeft', 'WristLeft'))

KIN_SEGMENTS = {'ForearmRight' : ('ElbowRight', 'WristRight'),
             'UpperarmRight' : ('ShoulderRight', 'ElbowRight'),
             'ForearmLeft' : ('ElbowLeft', 'WristLeft'),
             'UpperarmLeft' : ('ShoulderLeft', 'ElbowLeft'),
             'LowerlegRight' : ('KneeRight', 'AnkleRight'),
             'UpperlegRight' : ('HipRight', 'KneeRight'),
             'LowerlegLeft' : ('KneeLeft', 'AnkleLeft'),
             'UpperlegLeft' : ('HipLeft', 'KneeLeft')}

KIN_TREE = [('HipCenter','HipCenter'),('HipCenter', 'Spine'), ('HipCenter', 'HipLeft'), ('HipCenter', 'HipRight'),
            ('Spine', 'ShoulderCenter'), ('HipLeft', 'KneeLeft'), ('HipRight', 'KneeRight'),
            ('ShoulderCenter', 'ShoulderLeft'), ('ShoulderCenter', 'Head'), ('ShoulderCenter', 'ShoulderRight'),
            ('KneeLeft', 'AnkleLeft'), ('KneeRight', 'AnkleRight'), ('ShoulderLeft', 'ElbowLeft'), 
            ('ShoulderRight', 'ElbowRight'),
            ('AnkleLeft', 'FootLeft'), ('AnkleRight', 'FootRight'), ('ElbowLeft', 'WristLeft'),
            ('ElbowRight', 'WristRight'),
            ('WristLeft', 'HandLeft'), ('WristRight', 'HandRight')]

DEFAULT_LENGTHS = {('HipCenter','HipCenter'): 0,
                   ('HipCenter', 'Spine'): 0.3,
                   ('HipCenter', 'HipLeft'): 0.3,
                   ('HipCenter', 'HipRight'): 0.3,
                   ('Spine', 'ShoulderCenter'): 0.7,
                   ('HipLeft', 'KneeLeft'): 0.7,
                   ('HipRight', 'KneeRight'): 0.7,
                   ('ShoulderCenter', 'ShoulderLeft'): 0.3,
                   ('ShoulderCenter', 'Head'): 0.3,
                   ('ShoulderCenter', 'ShoulderRight'): 0.3,
                   ('KneeLeft', 'AnkleLeft'): 0.5,
                   ('KneeRight', 'AnkleRight'): 0.5,
                   ('ShoulderLeft', 'ElbowLeft'): 0.5,
                   ('ShoulderRight', 'ElbowRight'): 0.5,
                   ('AnkleLeft', 'FootLeft'): 0.1,
                   ('AnkleRight', 'FootRight'): 0.1,
                   ('ElbowLeft', 'WristLeft'): 0.5,
                   ('ElbowRight', 'WristRight'): 0.5,
                   ('WristLeft', 'HandLeft'): 0.1,
                   ('WristRight', 'HandRight'): 0.1
                  }


EXTENDED_KIN_LABELS = ['HipCenter', 'Spine', 'ShoulderCenter', 'Head',
              'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
              'ShoulderRight', 'ElbowRight', 'WristRight',
              'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft',
              'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight',
              'FootRight', ('HipCenter', 'HipCenter'), ('HipCenter', 'Spine'), ('HipCenter', 'HipLeft'), ('HipCenter', 'HipRight'),
            ('Spine', 'ShoulderCenter'), ('HipLeft', 'KneeLeft'), ('HipRight', 'KneeRight'),
            ('ShoulderCenter', 'ShoulderLeft'), ('ShoulderCenter', 'Head'), ('ShoulderCenter', 'ShoulderRight'),
            ('KneeLeft', 'AnkleLeft'), ('KneeRight', 'AnkleRight'), ('ShoulderLeft', 'ElbowLeft'), 
            ('ShoulderRight', 'ElbowRight'),
            ('AnkleLeft', 'FootLeft'), ('AnkleRight', 'FootRight'), ('ElbowLeft', 'WristLeft'),
            ('ElbowRight', 'WristRight'),
            ('WristLeft', 'HandLeft'), ('WristRight', 'HandRight')]

#ANGLES = [(limb[0], limb[1], limb2[1]) for limb in KIN_TREE for limb2 in KIN_TREE if limb[1] is limb2[0]]

ANGLES = [#('HipCenter', 'Spine', 'ShoulderCenter'), 
          #('HipCenter', 'HipLeft', 'KneeLeft'),
          #('HipCenter', 'HipRight', 'KneeRight'),
          #('Spine', 'ShoulderCenter', 'ShoulderLeft'), 
          #('Spine', 'ShoulderCenter', 'Head'),
          #('Spine', 'ShoulderCenter', 'ShoulderRight'),
          ('HipLeft', 'KneeLeft', 'AnkleLeft'), 
          ('HipRight', 'KneeRight', 'AnkleRight'),
          #('ShoulderCenter', 'ShoulderLeft', 'ElbowLeft'),
          #('ShoulderCenter', 'ShoulderRight', 'ElbowRight'), 
          ('KneeLeft', 'AnkleLeft', 'FootLeft'),
          ('KneeRight', 'AnkleRight', 'FootRight'),
          ('ShoulderLeft', 'ElbowLeft', 'WristLeft'), 
          ('ShoulderRight', 'ElbowRight', 'WristRight')#,
          #('ElbowLeft', 'WristLeft', 'HandLeft'),
          #('ElbowRight', 'WristRight', 'HandRight')
          ]

KIN_GRAPH = dict()
for label in KIN_LABELS:
    KIN_GRAPH[label] = []
    for seg in KIN_TREE:
        if seg[0] is label: KIN_GRAPH[label].append(seg[1])

MAPPING = {'WristRight' : 'Rw',
           'WristLeft' : 'Lw',
           'ShoulderRight' : 'Rsh',
           'ShoulderLeft' : 'Lsh',
           'ElbowRight' : 'Reb',
           'ElbowLeft' : 'Leb',
           'KneeRight' : 'Rknee',
           'KneeLeft' : 'Lknee',
           'AnkleRight' : 'Rank',
           'AnkleLeft' : 'Lank',
           'HipLeft' : 'Lhip',
           'HipRight' : 'Rhip',
           'HipCenter' : None,
           'Spine' : None,
           'ShoulderCenter' : None,
           'FootRight' : 'Rank', # use ankles as feet for now
           'FootLeft' : 'Lank',
           'Head' : None,
           'HandRight' : 'Rw', # use wrists as hands for now
           'HandLeft' : 'Lw'}

BONE_MAPPING = {#not sure what to do with lumbar and thorax for now. hm.
                # thorax, sacrum, lumbar given relative to world (qualysis x,y,z)
                # coord systems there are: z up spine, x out of body, y pointing left
                #### start unclear region
                ('HipCenter', 'HipCenter'): 'Thorax', # ROOT OF TREE, ABSOLUTE QUATERNION
                ('HipCenter', 'Spine'): 'Thorax',
                ('HipCenter', 'HipLeft'): 'Lumbar',
                ('HipCenter', 'HipRight'): 'Lumbar',
                ('Spine', 'ShoulderCenter'): 'Thorax',
                #### end unclear region
                ('HipLeft', 'KneeLeft'): 'Left Thigh',
                ('HipRight', 'KneeRight'): 'Right Thigh',
                #('ShoulderCenter', 'ShoulderLeft'): 'Left Upper Arm', # missing: get from Kevin
                #('ShoulderCenter', 'ShoulderRight'): 'Right Upper Arm', # missing: get from Kevin
                ('KneeLeft', 'AnkleLeft'): 'Left Shank',
                ('KneeRight', 'AnkleRight'): 'Right Shank',
                ('ShoulderLeft', 'ElbowLeft'): 'Left Upper Arm',
                ('ShoulderRight', 'ElbowRight'): 'Right Upper Arm',
                ('AnkleLeft', 'FootLeft'): 'Left Foot',
                ('AnkleRight', 'FootRight'): 'Right Foot',
                ('ElbowLeft', 'WristLeft'): 'Left Forearm',
                ('ElbowRight', 'WristRight'): 'Right Forearm'
}

CONVERSION_FACTOR = 1.0 #exports now in meters instead of mm

TILT = -14.0/180.0 * np.pi

# matrix to fix the -14 tilt angle, working in coordinates after permutation
# transposed because needs right mult
ROT = np.array([[np.cos(TILT), 0, -1*np.sin(TILT)], [0,1,0], [np.sin(TILT), 0, np.cos(TILT)]]).T 

class Error(Exception):
    pass

class KindError(Error):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LabelError(Error):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def expand_qual_label(label):
    if label in QUAL_LABELS:
        return (label + "X", label + "Y", label + "Z")
    else:
        return expand_kin_label(label)

def qual_quat_name(s):
    try:
        parts = s.split("/")
        label = parts[0].strip() + parts[-2].strip()
    except:
        label = s
    return label

def expand_qual_quat(bone):
    label = BONE_MAPPING[bone]
    return (label + "Q0", label + "Q1", label + "Q2", label + "Q3")

def expand_kin_label(label):
    return (label + "X", label + "Y", label + "Z")

def expand_kin_quat(bone):
    # ShoulderCenter xy-plane approximates
    # body plane for abduction, etc., in regression against qualysis
    return (bone[1] + "Q0", bone[1] + "Q1", bone[1] + "Q2", bone[1] + "Q3")

def kin_bone_name(bone):
    return bone[1] + "Q"

def clean_model(DM):
    """Break up a DataModel into a list of models according
    to missing data"""
    models = []
    
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


class DataObject(object):
    """Base class with raw and smooth data dictionaries"""
    def __init__(self):
        self.raw = {}
        self.smooth = {}
        self.unfiltered = {} # probably only used in kinect data

class Model(object):
    def dump_data(self, kind='raw'):
        labels = []
        for label in EXTENDED_KIN_LABELS:
            if type(label) == tuple:
                labels.append(kin_bone_name(label))
            else:
                labels.append(label)
        labels.append('Time')
        dalist = {}
        for label in labels:
            try:
                dalist[label] = getattr(self.data, kind)[label].tolist()
            except KeyError:
                pass
        return dalist

class KinModel(Model):
    def __init__(self, file_name, coord="old"):
        self.input_data = pd.io.parsers.read_csv(file_name)
        self.raw_input_data = pd.io.parsers.read_csv(file_name) # keep a copy that resists constrain_input_times
        self.data = DataObject()
        self.clean_and_filter(coord=coord)

    def clean_and_filter(self, coord):
        self.process_input(coord)
        # savitzsky-golay smooth
        self.sg_filter_data()
        #self.smoothing = 'SG'
        # make butterworth params, assuming ultimately resampled at 60
        b, a = signal.butter(ORDER, FREQ_CUTOFF)
        # low-pass butterworth
        for label in KIN_LABELS:
            self.butterworth_filter(label, b, a)

    def constrain_input_times(self, times=None):
        if times == None:
            return
        ticks = self.input_data['Ticks']
        min_tick = times[0] * TIME_SCALE + ticks[0]
        max_tick = times[-1] * TIME_SCALE + ticks[0]
        self.input_data = self.input_data[(ticks >= min_tick) & (ticks <= max_tick)]
        self.clean_and_filter()

    def constrain_times(self, times=None):
        if times == None:
            return
        ticks = self.input_data['Ticks']
        min_tick = times[0] * TIME_SCALE + ticks[0]
        max_tick = times[-1] * TIME_SCALE + ticks[0]
        for key in self.data.raw:
            self.data.raw[key] = np.array(self.data.raw[key])[(ticks >= min_tick) & (ticks <= max_tick)]
        #self.clean_and_filter()


    def process_input(self, coord="old"):
        # TIME_SCALE makes sure things are in ms
        ticks = np.array(self.input_data['Ticks'])
        self.data.raw['Time'] = (ticks - ticks[0]) / TIME_SCALE
        for label in KIN_LABELS:
            ex = expand_kin_label(label)
            # adjust coordinates and assemble per-label-positions
            # NEW: coordinates are adjusted on recording
            # old: ex[2], ex[0], ex[1]; new 0, 1, 2
            cord = [2, 0, 1] if coord == "old" else [0, 1, 2]
            try:
                self.data.raw[label] = (np.column_stack((self.input_data[ex[cord[0]]],
                                                        self.input_data[ex[cord[1]]],
                                                        self.input_data[ex[cord[2]]]))).dot(ROT)
                # keep an unfiltered copy
                self.data.unfiltered[label] = (np.column_stack((self.input_data[ex[cord[0]]],
                                                        self.input_data[ex[cord[1]]],
                                                        self.input_data[ex[cord[2]]]))).dot(ROT)

            except KeyError:
                pass


    def filter_data(self):
        joint_smoother = quaternions.PositionSmoother()
#        quat_smoother = quaternions.QuatSmoother()

        self.smoothing = "DE"        
        
        for label in KIN_LABELS:
            self.data.raw[label] = joint_smoother.smooth(self.data.raw[label])
            
#        for bone in KIN_TREE:
#            self.data.raw[kin_bone_name(bone)] = quat_smoother.smooth(self.data.raw[kin_bone_name(bone)])

    def dampen_jitter(self, label, alpha=0.1):
        from utilities import norm
        # assumed at 33 per second
        THRESHOLD = 1.0 # meters per second speed of joint
        series = np.array(self.data.raw[label])
        times = self.data.raw['Time']
        newseries = []
        newseries.append(series[0])
        for i in range(1, len(series)):
            if np.sqrt(((series[i] - series[i - 1])**2).sum(-1)) / (times[i] - times[i - 1]) > THRESHOLD:
                newseries.append((1 - alpha) * newseries[i - 1] + alpha * series[i])
            else:
                newseries.append(series[i])
        self.data.raw[label] = np.array(newseries)

    def butterworth_filter(self, label, b, a):
        series = np.array(self.data.raw[label])
        times = self.data.raw['Time']
        x = UnivariateSpline(times, series.T[0], s=0)
        y = UnivariateSpline(times, series.T[1], s=0)
        z = UnivariateSpline(times, series.T[2], s=0)
        newtimes = np.arange(times[0], times[-1], 1.0 / RATE)

        filt_x = signal.filtfilt(b, a, x(newtimes))
        filt_y = signal.filtfilt(b, a, y(newtimes))
        filt_z = signal.filtfilt(b, a, z(newtimes))

        new_x = UnivariateSpline(newtimes, filt_x, s=0)
        new_y = UnivariateSpline(newtimes, filt_y, s=0)
        new_z = UnivariateSpline(newtimes, filt_z, s=0)

        self.data.raw[label] = np.column_stack((new_x(times), new_y(times), new_z(times)))

    def sg_filter_data(self):
        window = 33
        degree = 3
        times = self.data.raw['Time']
        newtimes = np.linspace(times[0], times[-1], 33*(times[-1] - times[0]))
        for label in KIN_LABELS:
            # if not 'Hand' in label and not 'Foot' in label:
            #     self.dampen_jitter(label)
            filtered = []
            for i in range(3):
                temp = UnivariateSpline(times, self.data.raw[label].T[i], s=0)(newtimes)
                filt = sav_gol.savgol_filter(temp, window, degree)
                filt_spline = UnivariateSpline(newtimes, filt, s=0)
                filtered.append(filt_spline(times))
            self.data.raw[label] = np.column_stack((filtered[i] for i in range(3)))
        self.smoothing = "SG"


    def smooth_data(self, s=0):
        times = self.data.raw['Time']
        labels = KIN_LABELS
        self.data.smooth = dict()
        self.data.smooth['Time'] = times # how did I miss this before?!
        self.data.smoothers = dict()
        for label in labels:
            r = range(3) #range(len(self.data.raw[label][0]))
            f = [UnivariateSpline(times, self.data.raw[label][:, i], s=s) for i in r]
            self.data.smoothers[label] = f
            try:
                self.data.smooth[label] = np.column_stack((f[i](times) for i in r))
            except AttributeError:
                self.data.smooth = dict()
                self.data.smooth[label] = np.column_stack((f[i](times) for i in r))

    # unclean dumps for now: no masking of bad data points
    # dump everything to json, so browser model can interact with all data
    # including marking bad points by hand
    def constrain_label(self, label, times):
        raw = self.data.raw[label]
        constrained = [raw[i] for i in len(raw)
                       if self.data.raw['Time'][i] in times]
        return constrained


    def plot(self, label, coord, kind='raw'):
        try:
            data = getattr(self.data, kind)[label]
            pylab.plot(self.data.raw['Time'], data[:, coord])
        except KeyError:
            pass # improve this


class QualModel(Model):
    # ever need to also import raw data, check for missing values/bad times?
    # could use this for a mask -- just exclude bad times outright.
    # new data sets look much better
    # looks like default header value for Kevin's exports is 7
    def __init__(self, file_name=None, file_header=HEADER, quat_file_name=None, quat_header=HEADER):
        self.input_data = pd.io.parsers.read_csv(file_name,
                                                 header=file_header)
        self.raw_input_data = pd.io.parsers.read_csv(file_name,
                                                 header=file_header)

        if quat_file_name:
            self.input_quats = pd.io.parsers.read_csv(quat_file_name,
                                                      header=quat_header)
            # try to strip crap out of input and just keep mapped bone names
            self.input_quats = self.input_quats.rename(columns=qual_quat_name)#, inplace=True)
            self.raw_input_quats = pd.io.parsers.read_csv(quat_file_name,
                                                      header=quat_header)
            # try to strip crap out of input and just keep mapped bone names
            self.raw_input_quats = self.raw_input_quats.rename(columns=qual_quat_name)#, inplace=True)
        else:
            self.input_quats = None

        self.data = DataObject()
        self.process_input()

    def constrain_input_times(self, times=None):
        if times == None:
            return
        frames = self.input_data['Frame #']
        min_frame = times[0] * 60.0 + frames[0]
        max_frame = times[-1] * 60.0 + frames[0]
        self.input_data = self.input_data[(frames >= min_frame) & (frames <= max_frame)]
        try:
            self.input_quats = self.input_quats[(frames >= min_frame) & (frames <= max_frame)]
        except: # what kind of error would go here?
            pass
        self.process_input()

    def process_input(self):
        # incoming at 60 Hz, export files contain frame numbers, not times
        self.data.raw['Time'] = np.array(self.input_data['Frame #'])/60.0

        for label in QUAL_LABELS:
            ex = expand_qual_label(label)
            # assemble per-label positions
            try:
                self.data.raw[label] = CONVERSION_FACTOR * np.column_stack((self.input_data[ex[0]],
                                                        self.input_data[ex[1]],
                                                        self.input_data[ex[2]]))
                # mask 0s. in theory this should carry elsewhere? be careful!
                np.ma.masked_values(self.data.raw[label], 0)
                
            except KeyError:
                #hm. still necessary?
                print "Problem with %s"%label
                pass

        for bone in KIN_TREE:
            # careful of weird bones in the mapping
            if bone in BONE_MAPPING:
                ex = expand_qual_quat(bone)
                self.data.raw[kin_bone_name(bone)] = np.column_stack((self.input_quats[ex[0]],
                                                       self.input_quats[ex[1]],
                                                       self.input_quats[ex[2]],
                                                       self.input_quats[ex[3]]))
        
        # eventually will need to use the raw data here
        # and just do a crude missing data measure: throw it all away if
        # part is missing. I guess. (Or keep track of how model is constructed in the sense
        # of joint mappings; don't try to reconstruct model, just use that mapping to throw 
        # away missing data.)
        self.missing_data = dict()
        self.missing_time = dict()
        for label in QUAL_LABELS:
            try:
                self.missing_data[label] = np.array(self.data.raw[label] == 0)
                self.missing_time[label] = interp1d(self.data.raw['Time'], 0.3333*self.missing_data[label].sum(1), kind='linear', fill_value=1)
            except KeyError:
                print "Problem with %s" % label
                pass
        missing = np.column_stack((np.array(self.missing_data[label]).sum(1) for label in self.missing_data)).sum(1)
        self.missing_time_support = interp1d(self.data.raw['Time'], missing, kind='linear')

        if self.data.raw is not None:
            # add mapped kinect info to raw_qual_data dict
            for label in MAPPING:
                if MAPPING[label] is not None:
                    self.data.raw[label] = self.data.raw[MAPPING[label]]
                    self.missing_time[label] = self.missing_time[MAPPING[label]]

    def smooth_data(self, s=0, times=None):
        # for now: never use non-None times! will prob remove this
        if times == None:
            times = self.data.raw['Time']
        labels = []
        for label in BASIC_QUAL_LABELS:
           labels.append(label) # ALSO TAKE BONES HERE TO DO QUATERNIONS?
        labels.extend([kin_bone_name(bone) for bone in KIN_TREE if bone in BONE_MAPPING])
       # if hasattr(self, 'data.smooth'): return
        self.data.smooth = dict()
        self.data.smoothers = dict()
        for label in labels:  
            r = range(len(self.data.raw[label][0]))
            try:
                mask = (self.data.raw[label] == 0).sum(1)
                f = [UnivariateSpline(self.data.raw['Time'][mask == 0], 
                                      self.data.raw[label][mask == 0, i], 
                                      s=s) for i in r]
                self.data.smooth[label] = np.column_stack((f[i](times) for i in r))
                self.data.smoothers[label] = f
            except KeyError:
                pass
        
        self.data.smooth['Time'] = times

        for label in MAPPING: # NEED TO FIGURE OUT WHAT DO WITH QUATERNIONS HERE, TOO; SHOULD BE OK WITH LABELS ABOVE
            if MAPPING[label] is not None:
                self.data.smooth[label] = self.data.smooth[MAPPING[label]]
                self.data.smoothers[label] = self.data.smoothers[MAPPING[label]]


    def dump_smooth_data(self):
        return self.dump_data(kind='smooth')

    def plot(self, label, coord, kind="raw"):
        try:
            data = getattr(self.data, kind)[label]
            pylab.plot(self.data.raw['Time'], data[:, coord])
        except KeyError:
            pass # improve this

def dump_data(dic):
    dalist = {label: dic[label].tolist() for label in dic}
    return dalist


class DataModel(object):
    # has raw data
    # other classes should be able to attach properties
    # like specific limbs, etc.?
    # model is presented as columns with consecutive
    # triples representing the three coordinates of a joint
    # NOTE: kinect reverses the meaning of Z and Y. Hm.

    def __init__(self, kin_file_name=None, 
                 qual_file_name=None, 
                 qual_quat_file_name=None, 
                 qual_file_header=HEADER, 
                 qual_quat_header=HEADER): #header looks like 7 in new exports
        self.time_shift = 0
        self.alignment = None
        if qual_file_name is not None and qual_quat_file_name is not None:
            self.qual = QualModel(file_name=qual_file_name, file_header=qual_file_header, quat_file_name=qual_quat_file_name, quat_header=qual_quat_header)
        if kin_file_name is not None:
            self.kin = KinModel(kin_file_name)

    def smooth_data(self):
        self.kin.smooth_data()
        self.qual.smooth_data()

    def dump_data(self):
        kin = self.kin.dump_data() if hasattr(self, "kin") else None
        flags = np.zeros(len(self.kin.data.raw['Time'])).tolist() if hasattr(self, "kin") else None
        qual = self.qual.dump_data() if hasattr(self, "qual") else None
        qual_smooth = self.qual.dump_smooth_data() if hasattr(self, "qual") else None
        
        to_dump = {'kin': kin,
                   'qual': qual,
                   'qual_smooth': qual_smooth,
                   'time_shift': self.time_shift,
                   'flags': flags}
        return json.dumps(to_dump)
        
    
    def loss(self, time=0, radians=None, intercept=None):
        """times: index into the time array"""
        REFS = ['HipRight', 'HipLeft','ShoulderRight', 'ShoulderLeft', 'ElbowRight', 'WristRight', 'ElbowLeft', 'WristLeft', 'KneeRight', 'KneeLeft', 'AnkleRight', 'AnkleLeft']
        m = rot_matrix(radians)
        q = np.vstack((np.array(self.qual.data.smooth[ref])[time:time+33].dot(m.T) + intercept for ref in REFS))
        k = np.vstack((np.array(self.kin.data.raw[ref])[time:time+33] for ref in REFS))
        
        return ((q-k)*(q-k)).ravel().sum()
        
    def min_loss(self, arg):
        alpha, a, b, c = tuple(arg)
        N = len(self.kin.data.raw['Time']) / 33
        loss = -1
        for n in range(N):
            newloss = self.loss(33 * n, radians=alpha, intercept=np.array([a,b,c]))
            if loss < 0: 
                loss = newloss
            elif newloss < loss:
                loss = newloss
        return loss    
                
    
    def compute_affine(self):
        """Use linear regression and first few seconds of data to get
        best affine coordinate transformation between kinect and qualysis.
        Only call this on synchronized models with smoothed qualysis data 
        evaluated on kinect timestamps (e.g., as in total_model) """
        from scipy.optimize import minimize
        minimizer = minimize(self.min_loss, [0,0,0,0]).x
        self.alignment = minimizer
        return rot_matrix(minimizer[0]), minimizer[1:]

    def align(self):
        import quaternions
        radians, a, b, c = self.alignment
        m = rot_matrix(radians)
        intercept = np.array([a, b, c])
        for label in KIN_LABELS:
            try:
                self.qual.data.smooth[label] = self.qual.data.smooth[label].dot(m.T) + intercept
            except KeyError:
                pass
        # replace HipCenterQ by canonical form eventually
        self.qual.data.smooth['HipCenterQ'] = quaternions.conjugate(
                                                np.array([np.cos(0.5*radians), 
                                                          0, 0, 
                                                          np.sin(0.5*radians)]), 
                                                self.qual.data.smooth['HipCenterQ'])
        
def rot_matrix(radians):
    return np.array([[np.cos(radians), -np.sin(radians), 0], 
          [np.sin(radians), np.cos(radians), 0], 
          [0, 0, 1]])        

if __name__ == '__main__':
    name = "mr"    
    kin_path = "DataFiles/kin/" + name + ".csv"
    qual_quat_path = "DataFiles/qual/" + name + "q.csv"
    qual_joint_path = "DataFiles/qual/" + name + "c.csv"
    DM = DataModel(kin_file_name=kin_path, qual_file_name=qual_joint_path, qual_quat_file_name=qual_quat_path)
    DM.smooth_data()