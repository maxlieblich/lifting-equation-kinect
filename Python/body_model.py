import numpy as np
import numpy.ma as ma
import data_model
from utilities import norm, normalize_rows
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import sav_gol
import quaternions as q
from scipy import optimize
from collections import OrderedDict

LIMBS = [('ShoulderRight', 'ElbowRight', 'WristRight'),
         ('ShoulderLeft', 'ElbowLeft', 'WristLeft'),
         ('HipRight', 'KneeRight', 'AnkleRight'),
         ('HipLeft', 'KneeLeft', 'AnkleLeft')]

# not meant for qualysis data
def extract_lengths(data):
    lengths = dict()
    for seg in data_model.KIN_TREE:
        # for now, stupidly average over whole sample
        try:
            lengths[seg] = np.mean(norm(np.array(data[seg[1]]) - np.array(data[seg[0]])))
        except KeyError:
            pass
    return lengths


def fit_to_sphere(point, radius=1):
    # fit one 3-dim vector to a sphere
    # applies to numpy array with 3d rows
    # freaks out when point is 0
    try:
        fitted = radius * point / norm(point)
    except:
        fitted = 0
    return fitted

def new_coords(point):
    # point is dictionary with coords for kinect joints
    # or possibly array of such points -- DM.raw_kin_data (without Time), for example
    newpoint = dict()
    newpoint['HipCenter'] = point['HipCenter']
    for seg in data_model.KIN_TREE:
        newpoint[seg] = np.array(point[seg[1]]) - np.array(point[seg[0]])
    other_keys = [label for label in point if label not in data_model.KIN_LABELS]
    for label in other_keys:
        newpoint[label] = point[label]
    return newpoint

def old_coords(point):
    # HipCenter, then everything else is relative
    oldpoint = dict()
    oldpoint['HipCenter'] = point['HipCenter']
    visited = ['HipCenter']
    change = 1
    while change > 0:
        length = len(visited)
        for edge in data_model.KIN_TREE:
            if edge[0] in visited and edge[1] not in visited: 
                oldpoint[edge[1]] = point[edge] + oldpoint[edge[0]]
                visited.append(edge[1])
        change = len(visited) - length
    other_keys = [label for label in point if label not in data_model.KIN_TREE]
    for label in other_keys:
        oldpoint[label] = point[label]
    return oldpoint


def triangle_min(L1, L2, a, b, c):
    # vertices are (0,0), (b, c), (a, 0)
    # returns pair of angles from horizontal for two successive legs, starting at (0,0)
    # with given lengths
    cost = lambda X: X[2]**2 + X[3]**2 + (X[2] + L1 * np.cos(X[0]) - b)**2 + (X[3] + L1 * np.sin(X[0]) - c)**2 + (X[2] + L1 * np.cos(X[0]) + L2 * np.cos(X[1]) - a)**2 + (X[3] + L1 * np.sin(X[0]) + L2 * np.sin(X[1]))**2
    soln = minimize(cost, [0,0,0,0]).x
    A = np.array([soln[2], soln[3]])
    B = A + np.array((L1 * np.cos(soln[0]), L1 * np.sin(soln[0])))
    C = B + np.array((L2 * np.cos(soln[1]), L2 * np.sin(soln[1])))
    return (A, B, C)

def eff_min(L1, L2, a, b, c):
    # vertices are (0,0), (b, c), (a, 0)
    # returns pair of angles from horizontal for two successive legs, starting at (0,0)
    # with given lengths
    cost = lambda X: (L1 * np.cos(X[0]) + L2 * np.cos(X[1]) - a)**2 + (L1 * np.sin(X[0]) + L2 * np.sin(X[1]))**2
    soln = minimize(cost, [0,0,0,0]).x
    A = np.array([soln[2], soln[3]])
    B = A + np.array((L1 * np.cos(soln[0]), L1 * np.sin(soln[0])))
    C = B + np.array((L2 * np.cos(soln[1]), L2 * np.sin(soln[1])))
    return (A, B, C)


def sg_filter_data(point):
    window = 33
    degree = 3
    if 'Time' not in point:
        raise Exception('Need time to filter!')
    times = point['Time']
    newtimes = np.linspace(times[0], times[-1], 33 * (times[-1] - times[0]))
    outpoint = {}
    for label in point:
        if not label in data_model.KIN_TREE:
            outpoint[label] = point[label]
        else:
            filtered = []
            for i in range(3):
                temp = UnivariateSpline(times, point[label].T[i], s=0)(newtimes)
                filt = sav_gol.savgol_filter(temp, window, degree)
                filt_spline = UnivariateSpline(newtimes, filt, s=0)
                filtered.append(filt_spline(times))
            outpoint[label] = np.column_stack((filtered[i] for i in range(3)))
    return outpoint

# IN NEUTRAL POSITION: X POINTS FORWARD, Y POINTS ALONG CHILD BONES

# assume that the "child segment" is aligned with positive z-axis
# and has a given length
# in the following two classes
# a joint is an orthogonal transformation (change of local coord systems); the translations
# are built into the chain, not the joint.
# every joint has incoming and outgoing bones (?), as defined by ordering in chain sequence

# TRANSFORM METHODS GIVE TRANSFORMATIONS FROM CHILD COORDINATES (BONE ENDING AT JOINTCHILD) TO PARENT COORDINATES

class HingeJointChild(object):
    def __init__(self, name, limit=[np.pi], angle=0):
        self.name = name
        self.limit = limit  # numerical limit on angle; likely to be [0, 150] degrees or so?
        self.angle = angle # initial angle of child relative to parent for this joint

    def update(self, x):
        self.angle = x

    def transform(self):
        # return quaternion giving coord trans
        # pay attention to vectorization; this might be weird for single rows?
        return q.invert(np.column_stack((np.cos(0.5 * self.angle), 0, -1.0 * np.sin(0.5 * self.angle), 0)))  # rotate by negative angle

# pole is (0,0,-1) for this one
def stereographic(a, b, sign=1.0):
    return np.array([4 * a / (a**2 + b**2 + 4), 4 * b / (a**2 + b**2 + 4), sign * (4 - a**2 - b**2) / (a**2 + b**2 + 4)])


def destereo(v, sign=1.0):
    # v: unit vector
    return np.array([2 * v[0] / (sign * v[2] + 1), 2 * v[1] / (sign * v[2] + 1)])

class BallJointChild(object):
    def __init__(self, name, limits=[0, 0], direction=np.array((0.01, 0, 1)), axial=0, sign=1.0):
        self.name = name
        self.limits = limits
        #  approx: equation for cone ax^2 + bz^2 >= y^2 constraining motion; outside certain cone.
        self.direction = normalize_rows(direction)  # point on 2-sphere giving (initial) direction of child relative to parent
        #self.phi = 0 # rotation around x
        #self.theta = 0 # rotation around z
        self.axial = axial  # rotation around bone axis
        self.sign = sign

    def update(self, x):
        self.direction = stereographic(*x[:2], sign=self.sign)
        self.axial = x[2]

    def transform(self):
        # quaternion realizing transformation
        q1 = np.array((np.cos(0.5 * self.axial), 0, 0, np.sin(0.5 * self.axial)))
        q2 = q.axis_mover(self.direction)
        return q.invert(q.prod(q1, q2))


class KinematicUnit(object):
    def __init__(self):
        self.labels = []

    def get_joint(self, name):
        self.joint_positions()
        names = [label.name for label in self.labels]
        try:
            i = names.index(name)
            return self.positions[i]
        except:
            pass  # hm.


class HipJoint(object):
    def __init__(self, name):
        self.name = name

class HipTree(KinematicUnit):
# assume hip unit is rigid with 90 degree angle
# and 135 degrees between spine and hip segments
# and same length on both sides
# default orientation: horizontal
# model this as scaled version of right triangle with sides of length 1
    def __init__(self, lengths=[1, 1]):
        self.labels = [HipJoint('HipCenter'), HipJoint('HipLeft'), HipJoint('HipRight'), HipJoint('Spine')]
        self.lengths = lengths  # find standard lengths?
        self.Spine = np.array((0, 0, 1))
        self.HipLeft = np.array((0, 1 / np.sqrt(2), -1 / np.sqrt(2)))
        self.HipRight = np.array((0, -1 / np.sqrt(2), -1 / np.sqrt(2)))
        self.HipCenter = np.array((0, 0, 0))
        self.orientation = np.array((1, 0, 0, 0))  # default: no action
        self.abs_orientations = [np.array((1, 0, 0, 0)) for i in range(3)]
        self.positions = [self.HipCenter, self.HipLeft, self.HipRight, self.Spine]

    def joint_positions(self):
        self.positions[0] = self.HipCenter
        self.positions[1] = q.act(self.orientation, self.lengths[0] * self.HipLeft) + self.HipCenter
        self.positions[2] = q.act(self.orientation, self.lengths[0] * self.HipRight) + self.HipCenter
        self.positions[3] = q.act(self.orientation, self.lengths[1] * self.Spine) + self.HipCenter
        self.abs_orientations = [self.orientation for i in range(3)]

    def get_joint(self, name):
        self.joint_positions()
        names = [label.name for label in self.labels]
        try:
            i = names.index(name)
            return self.positions[i]
        except:
            pass  # hm.


class KinematicChain(KinematicUnit):
    def __init__(self, labels, lengths, init_position=None):  # pass a Kinematic unit with a joint name preceding first label of chain
        self.labels = labels  # sequence of joints with types (BallJoint or HingeJoint)
        self.abs_orientations = []  # sequence of quats giving transformation to abs coordinates
        self.lengths = lengths  # sequence of lengths between successive joints, length one less!
        self.init_position = init_position
        self.positions = []
        self.joint_positions()  # need to call joint_positions at least once after filling in the labels and lengths

    def joint_positions(self):
        # displacements from the root, with default initial position (0,0,0)
        orienter = self.init_position.abs_orientations[-1]
        if len(self.positions) > 0:
            init_position = self.positions[0]  # if root is defined, keep it where it is
        else:
            if self.init_position is not None:
                init_position = self.init_position.get_joint(self.labels[0].name)  # initialize root position to end of previous
            else:
                init_position = np.array((0, 0, 0))  # or just put it at origin if no previous
        positions = [init_position]
       # print init_position
        self.abs_orientations = [orienter]
        for i in range(1, len(self.labels)):
            label = self.labels[i]
            self.abs_orientations.append(q.prod(self.abs_orientations[i - 1], label.transform()))
            new_end = positions[i - 1] + self.lengths[i - 1] * q.act(self.abs_orientations[i], np.array((0, 0, 1.0)))
            positions.append(new_end)
       # print positions
        self.positions = positions

    def get_joint(self, name):
        self.joint_positions()
        names = [label.name for label in self.labels]
        try:
            i = names.index(name)
            return self.positions[i]
        except:
            pass  # hm.


class KinematicModel(object):
    def __init__(self, units):
        self.units = units  #kinematic units: dictionary?

    def get_joint(self, name):
        for unit in self.units:
            if name in [label.name for label in self.units[unit].labels]:
                return self.units[unit].get_joint(name)


def chain_maker(obj, labels, init_position):
    # assume labels is a reasonable sequence of kinect joint names, and that obj has joints defined with those names,
    # AND that obj has a property "lengths" that has lengths indexed by the pairs making KIN_TREE;
    # no error checking for now!
    new_labels = [getattr(obj, label) for label in labels]
    new_lengths = [obj.lengths[(labels[i], labels[i + 1])] for i in range(len(labels) - 1)]
    return KinematicChain(new_labels, new_lengths, init_position)


class Human(KinematicModel):
    def __init__(self, body_model=None):
        if body_model:
            self.lengths = body_model.lengths
        else:
            self.lengths = data_model.DEFAULT_LENGTHS
        self.Spine = BallJointChild("Spine")
        self.ShoulderCenter = BallJointChild("ShoulderCenter", direction=np.array((0.01, 0, 1)))
        self.ShoulderRight = BallJointChild("ShoulderRight", direction=np.array((0, -1, -0.5)))  # attempting to get this right: rt shoulder rel to coord system coming from above choice
        self.ShoulderLeft = BallJointChild("ShoulderLeft", direction=np.array((0, 1, -0.5)))  # oh dear god, this is probably totally wrong. note: asym really comes from rt hand rule, not body!
        self.Head = BallJointChild("Head", direction=np.array((0.01, 0, 1))) # ends at kinect Head "joint", is really the coord trans to get head dir from shouldercenter
        self.ElbowRight = BallJointChild("ElbowRight", direction=np.array((-0.05, -1, 1)))
        self.ElbowLeft = BallJointChild("ElbowLeft", direction=np.array((-0.05, 1, 1)))
        self.WristRight = HingeJointChild("WristRight", angle=np.pi / 20)  # ends in hand
        self.WristLeft = HingeJointChild("WristLeft", angle=np.pi / 20)  # ends in hand
        self.HandRight = BallJointChild("HandRight", direction=np.array((0.01, 0, 1)))
        self.HandLeft = BallJointChild("HandLeft", direction=np.array((0.01, 0, 1)))
        self.HipRight = BallJointChild("HipRight")  # ball joint located at end of hipunit; assuming hipunit has one coord sys, no joints
        self.HipLeft = BallJointChild("HipLeft")  # ball joint located at end of hipunit
        self.KneeRight = BallJointChild("KneeRight", direction=np.array((0.01, 0, -1)), axial=0, sign=-1.0)
        self.KneeLeft = BallJointChild("KneeLeft", direction=np.array((0.01, 0, -1)), axial=0, sign=-1.0)
        self.AnkleRight = HingeJointChild("AnkleRight", angle=np.pi / 100)  # ends in foot
        self.AnkleLeft = HingeJointChild("AnkleLeft", angle=np.pi / 100)  # ends in foot
        self.FootRight = BallJointChild("FootRight",  direction=np.array((-1, 0, 0)))
        self.FootLeft = BallJointChild("FootLeft",  direction=np.array((-1, 0, 0)))
        self.Hips = HipTree([0.5 * (self.lengths[('HipCenter', 'HipRight')] + self.lengths[('HipCenter', 'HipLeft')]),
                             self.lengths[('HipCenter', 'Spine')]])
        self.units = OrderedDict()   # break body into chains, organized in dict with meeting points as in chain_maker
        self.units['Hips'] = self.Hips
        self.units['Backbone'] = chain_maker(self, ['Spine', 'ShoulderCenter'], self.units['Hips'])
        self.units['Neck'] = chain_maker(self, ['ShoulderCenter', 'Head'], self.units['Backbone'])
        self.units['ArmRight'] = chain_maker(self,
                                             ['ShoulderCenter', 'ShoulderRight', 'ElbowRight',
                                              'WristRight', 'HandRight'],
                                             self.units['Backbone'])
        self.units['ArmLeft'] = chain_maker(self,
                                            ['ShoulderCenter', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft'],
                                            self.units['Backbone'])
        self.units['LegRight'] = chain_maker(self, ['HipRight', 'KneeRight', 'AnkleRight', 'FootRight'],
                                             self.units['Hips'])
        self.units['LegLeft'] = chain_maker(self, ['HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft'],
                                            self.units['Hips'])

        self.reorient(np.array((0, 0, 0, 1)))   # rotate body so that it is facing -x direction: toward kinect

    # IN THE FOLLOWING:
        # USING STEREOGRAPHIC PROJECTION TO MAP SPHERE FOR NOW; THIS LEAVES OUT ONE POINT!
        # data structure defining input is an array like this:
        # x[0:3] hipcenter location
        # x[3:7] hipunit orientation quat
        # x[7:9], x[9] shoulercenter direction
        # x[10:12], x[12] shoulderright dir, axial
        # x[13:15], x[15] shoulderleft dir, axial
        # x[16:18] head direction; assume axial 0
        # x[18:20], x[20] elbowright
        # x[21:23], x[23] elbowleft
        # x[24] wristright
        # x[25] wristleft
        # x[26:28] handright, axial=0
        # x[28:30] handleft
        # x[30:32], x[32] kneeright
        # x[33:35], x[35] kneeleft
        # x[36] ankleright
        # x[37] ankleleft
        # x[38:40] footright, axial=0
        # x[40:42] footleft, axial=0

    def set_position(self):
        for u in self.units:
            self.units[u].joint_positions()

    def forward_kinematics(self, x):  # feed numpy ndarray
        self.Hips.HipCenter = x[:3]
        self.Hips.orientation = x[3:7]
        self.ShoulderCenter.update(x[7:10])
        self.ShoulderRight.update(x[10:13])
        self.ShoulderLeft.update(x[13:16])
        self.Head.direction = stereographic(*x[16:18])
        self.ElbowRight.update(x[18:21])
        self.ElbowLeft.update(x[21:24])
        self.WristRight.update(x[24])
        self.WristLeft.update(x[25])
        self.HandRight.direction = stereographic(*x[26:28])
        self.HandLeft.direction = stereographic(*x[28:30])
        self.KneeRight.update(x[30:33])
        self.KneeLeft.update(x[33:36])
        self.AnkleRight.update(x[36])
        self.AnkleLeft.update(x[37])
        self.FootRight.direction = stereographic(*x[38:40])
        self.FootLeft.direction = stereographic(*x[40:42])
        self.set_position()

    def reorient(self, quat):
        for u in self.units:
            self.units[u].abs_orientations = [q.prod(a, quat) for a in self.units[u].abs_orientations]
        self.set_position()

    chunk_segment = {'Hips': [0, 7],
         'ShoulderCenter': [7, 10],
         'ShoulderRight': [10, 13],
         'ShoulderLeft': [13, 16],
         'Head': [16, 18],
         'ElbowRight': [18, 21],
         'ElbowLeft': [21, 24],
         'WristRight': [24, 25],
         'WristLeft': [25, 26],
         'HandRight': [26, 28],
         'HandLeft': [28, 30],
         'KneeRight': [30, 33],
         'KneeLeft': [33, 36],
         'AnkleRight': [36, 37],
         'AnkleLeft': [37, 38],
         'FootRight': [38, 40],
         'FootLeft': [40, 42]}

    def encode_state(self):
        chunks = [self.Hips.HipCenter,
                  self.Hips.orientation,
                  destereo(self.ShoulderCenter.direction),
                  self.ShoulderCenter.axial,
                  destereo(self.ShoulderRight.direction),
                  self.ShoulderRight.axial,
                  destereo(self.ShoulderLeft.direction),
                  self.ShoulderLeft.axial,
                  destereo(self.Head.direction),
                  destereo(self.ElbowRight.direction),
                  self.ElbowRight.axial,
                  destereo(self.ElbowLeft.direction),
                  self.ElbowLeft.axial,
                  self.WristRight.angle,
                  self.WristLeft.angle,
                  destereo(self.HandRight.direction),
                  destereo(self.HandLeft.direction),
                  destereo(self.KneeRight.direction, self.KneeRight.sign),
                  self.KneeRight.axial,
                  destereo(self.KneeLeft.direction, self.KneeLeft.sign),
                  self.KneeLeft.axial,
                  self.AnkleRight.angle,
                  self.AnkleLeft.angle,
                  destereo(self.FootRight.direction),
                  destereo(self.FootLeft.direction)]
        return np.hstack(chunks)

    default_bounds = [
        (-5, 5), (-5, 5), (0, 5),                  # HipCenter
        (-1, 1), (-1, 1), (-1, 1), (-1, 1),        # HipOrientation
        (-np.pi, np.pi), (-np.pi, np.pi),          # ShoulderCenter
        (-np.pi, np.pi),
        (-np.pi, np.pi), (-np.pi, np.pi),          # ShoulderRight
        (-np.pi, np.pi),
        (-np.pi, np.pi), (-np.pi, np.pi),          # ShoulderLeft
        (-np.pi, np.pi),
        (-np.pi, np.pi), (-np.pi, np.pi),          # Head
        (-np.pi, np.pi), (-np.pi, np.pi),          # ElbowRight
        (-0.5 * np.pi, 0.5 * np.pi),
        (-np.pi, np.pi), (-np.pi, np.pi),          # ElbowLeft
        (-0.5 * np.pi, 0.5 * np.pi),
        (0, 0.75 * np.pi),                         # WristRight
        (0, 0.75 * np.pi),                         # WristLeft
        (-np.pi, np.pi), (-np.pi, np.pi),          # HandRight
        (-np.pi, np.pi), (-np.pi, np.pi),          # HandLeft
        (-np.pi, np.pi), (-np.pi, np.pi),          # KneeRight
        (-0.5 * np.pi, 0.5 * np.pi),
        (-np.pi, np.pi), (-np.pi, np.pi),          # KneeLeft
        (-0.5 * np.pi, 0.5 * np.pi),
        (0, 0.75 * np.pi),                         # AnkleRight
        (0, 0.75 * np.pi),                         # AnkleLeft
        (np.pi, 10), (-10, 10),                    # FootRight
        (0, 10), (-10, 10)                         # FootLeft
    ]

    def fit(self, datum, no_info=False, method='SLSQP', bounds=default_bounds):
        def distance(x, human):
            human.forward_kinematics(x)
            displacements = np.array([((human.get_joint(name) - np.array(datum[name]))**2).sum(-1) for name in data_model.KIN_LABELS])
            return (displacements**2).sum()
        if no_info:
            init = np.zeros(42)
        else:
            init = self.encode_state()
        res = optimize.minimize(distance, init, args=(self,), bounds=bounds, method=method)
        self.forward_kinematics(res.x)

    def forward_kinematics_joints(self, joints, x):
        y = self.encode_state()
        index = 0
        for i in range(len(joints)):  # read chunks out of x and put them in appropriate place in y
            ends = Human.chunk_segment[joints[i]]
            y[ends[0]:ends[1]] = x[index:index + ends[1] - ends[0]]
            index += ends[1] - ends[0]
        self.forward_kinematics(y)


    def forward_kinematics_unit(self, unit_name, x):
        u = self.units[unit_name]
        y = self.encode_state()
        if unit_name == 'Hips':
            y[:7] = x
            self.forward_kinematics(y)
        else:
            self.forward_kinematics_joints(self, [u.labels[label].name for label in u.labels][1:])

    def extract_joint_chunks(self, joints, y=None):
        if y is None:
            y = self.encode_state()
        x = []
        for i in range(len(joints)):  # read chunks out of y and append them to x
            ends = Human.chunk_segment[joints[i]]
            x.extend(y[ends[0]:ends[1]])
        return x

    def extract_unit_chunk(self, unit_name, y=None):
        u = self.units[unit_name]
        if y is None:
            y = self.encode_state()
        if unit_name == 'Hips':
            x = y[:7]
        else:
            x = self.extract_joint_chunks([u.labels[label].name for label in u.labels][1:], y=y)
        return x

    def extract_joint_bounds(self, joints, bounds=default_bounds):
        new_bounds = []
        for i in range(1, len(joints)):
            ends = Human.chunk_segment[joints[i]]
            new_bounds.extend(bounds[ends[0]:ends[1]])
        return new_bounds

    def extract_bounds(self, unit_name, bounds=default_bounds):
        u = self.units[unit_name]
        if unit_name == 'Hips':
            new_bounds = bounds[:7]
        else:
            new_bounds = self.extract_joint_bounds([u.labels[label].name for label in u.labels][1:], bounds=bounds)
        return new_bounds

    def fit_joints(self, joints, datum, no_info=False, method='SLSQP', bounds=default_bounds):
        def distance(x, human):
            human.forward_kinematics_joints(joints, x)
            displacements = np.array([((human.get_joint(name) - np.array(datum[name]))**2).sum(-1) for
                                      name in joints])
            return (displacements**2).sum()
        if no_info:
            init = np.zeros(42)
        else:
            init = self.encode_state()
        res = optimize.minimize(distance, self.extract_joint_chunks(joints, init), args=(self,),
                                bounds=self.extract_joint_bounds(joints, bounds), method=method)
        self.forward_kinematics_joints(joints, res.x)


    def fit_unit(self, unit_name, datum, no_info=False, method='SLSQP', bounds=default_bounds):
        def distance(x, human):
            human.forward_kinematics_unit(unit_name, x)
            displacements = np.array([((human.get_joint(name) - np.array(datum[name]))**2).sum(-1) for
                                      name in data_model.KIN_LABELS if hasattr(human.units[unit_name].labels, name)])
            return (displacements**2).sum()
        if no_info:
            init = np.zeros(42)
        else:
            init = self.encode_state()
        res = optimize.minimize(distance, self.extract_unit_chunk(unit_name, init), args=(self,), bounds=self.extract_bounds(unit_name, bounds), method=method)
        self.forward_kinematics_unit(unit_name, res.x)

    def fit_tree(self, datum, no_info=False, method='SLSQP', bounds=default_bounds):
        self.fit_unit('Hips', datum, no_info=no_info, method=method, bounds=bounds)
        parts = [['ShoulderCenter', 'ShoulderLeft', 'ShoulderRight', 'Head'],
                 ['ElbowRight', 'WristRight', 'HandRight'],
                 ['ElbowLeft', 'WristLeft', 'HandLeft'],
                 ['KneeRight', 'AnkleRight', 'FootRight'],
                 ['KneeLeft', 'AnkleLeft', 'FootLeft']]
        for part in parts:
            self.fit_joints(part, datum, no_info=no_info, method=method, bounds=bounds)



class LengthObject(object):
    __slots__ = ['lengths']

# model kinect body with tree of joints, to make coordinates
# for fitting model
class BodyModel(object):
    def __init__(self, data, standardized=False):
        self.data = data
        if standardized:
            self.lengths = data_model.DEFAULT_LENGTHS
        else:
            self.lengths = extract_lengths(data)
        size = np.array([self.lengths[key] for key in self.lengths]).sum()
        self.data['size'] = np.ones(len(self.data[self.data.keys()[0]])) * size

    def fit_point(self, point, filter=True):
        # need point to be point with kinect joint names
        #times = self.data['Time']
        newpoint = sg_filter_data(new_coords(point)) if filter else new_coords(point)
        fitted_point = dict()
        fitted_point['HipCenter'] = point['HipCenter']
        for seg in data_model.KIN_TREE:
            fitted_point[seg] = fit_to_sphere(newpoint[seg], self.lengths[seg])
        other_keys = [label for label in point if label not in data_model.KIN_TREE]
        for label in other_keys:
            fitted_point[label] = point[label]
        return old_coords(fitted_point)

    def greedy_fit_point(self, point, filter=True):      # point should be made of numpy arrays....
        new_point = sg_filter_data(point) if filter else point
        fitted_point = dict()
        fitted_point['HipCenter'] = new_point['HipCenter']
        visited = ['HipCenter']
        change = 1
        while change > 0:
            length = len(visited)
            for edge in data_model.KIN_TREE:
                if edge[0] in visited and edge[1] not in visited:
                    edge_length = self.lengths[edge]
                    dir = normalize_rows(np.array(new_point[edge[1]]) - np.array(fitted_point[edge[0]]))
                    fitted_point[edge[1]] = fitted_point[edge[0]] + edge_length * dir
                    visited.append(edge[1])
                   # print edge[1]
            change = len(visited) - length
        #other_keys = [label for label in new_point if label not in data_model.KIN_TREE]
        #for label in other_keys:
        #    fitted_point[label] = new_point[label]
        return fitted_point

    # planar fit seems good for now
    def fit_limbs(self):
        for limb in LIMBS:
            data = self.data
            lengths = self.lengths
            A = np.array(data[limb[0]])
            B = np.array(data[limb[1]])
            C = np.array(data[limb[2]])
            Y = normalize_rows(B - A)
            X = normalize_rows(C - A)
            Yn = normalize_rows(Y - np.sum((X * Y), axis=-1, keepdims=True) * X)
            a = ((C - A) * X).sum(-1)
            b = ((B - A) * X).sum(-1)
            c = ((B - A) * Yn).sum(-1)
            # will be dog slow for now
            temp = {}
            temp['start'] = []
            temp['mid'] = []
            temp['end'] = []
            L1 = self.lengths[(limb[0], limb[1])]
            L2 = self.lengths[(limb[1], limb[2])]
            L = len(A)
            for i in xrange(L):
                S, M, E = triangle_min(L1, L2, a[i], b[i], c[i])
                temp['start'].append(S)
                temp['mid'].append([M[0], M[1]])
                temp['end'].append([E[0], E[1]])
            temp['start'] = np.array(temp['start'])
            temp['end'] = np.array(temp['end'])
            temp['mid'] = np.array(temp['mid'])

            self.data[limb[0]] = A + (temp['start'].T[0]).reshape(L,1) * X + (temp['start'].T[1]).reshape(L,1) * Yn
            self.data[limb[1]] = A + (temp['mid'].T[0]).reshape(L,1) * X + (temp['mid'].T[1]).reshape(L,1) * Yn
            self.data[limb[2]] = A + (temp['end'].T[0]).reshape(L,1) * X + (temp['end'].T[1]).reshape(L,1) * Yn

    def scale(self):
        size = np.array([self.lengths[key] for key in self.lengths]).sum()
        for key in self.data:
            if key != 'Time':
                self.data[key] = (np.array(self.data[key]) / size).tolist()

    def fit(self, filter=True):
        self.data = self.fit_point(self.data, filter=filter)

    def greedy_fit(self, filter=True):
        self.data = self.greedy_fit_point(self.data, filter=filter)

