import math
import random
import data_model
import angle_handlers
import json
from functools import partial
import numpy as np
import pylab
from numpy.linalg import norm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import cross_validation
from utilities import normalize_rows
import cPickle as pickle
from sklearn.externals import joblib
import os
import multiprocessing

FUNCTIONS = [] # list of function objects
COORD_FUNCTIONS = []
TRAINING_DIR = "DataFiles/training/"

# use the DataFiles/master directory, import json that
# is now being exported by the position_pool flagged pools

def lookup(name):
    for func in FUNCTIONS:
        if func.name == name:
            return func
    for func in COORD_FUNCTIONS:
        if func.name == name: 
            return func
    return None

class Func(object):
    def __init__(self, func, name, limits=None):
        self.name = name
        self.func = func
        self.limits = limits

def joint_angle(joint, data):
    angle = [ang for ang in data_model.ANGLES if ang[1] == joint][0]
    fore = normalize_rows(np.array(data[angle[2]]) - np.array(data[angle[1]]))
    upper = normalize_rows(np.array(data[angle[0]]) - np.array(data[angle[1]]))
    meat = np.sum((fore * upper), axis=-1)
    return 180/math.pi * np.arccos(meat)

def quat_joint_angle(joint, data):
    angle = [ang for ang in data_model.ANGLES if ang[1] == joint][0]
    rawbone = (angle[1], angle[2])
    bone = data_model.kin_bone_name((angle[1], angle[2]))
    try:
        method = 'joint' if len(np.array(data[bone]).shape) == 1 else 'quat'
    except:
        method = 'joint'
    # assuming angle between z axes, which seems to work....
    raw_func = angle_handlers.z_angle(data=data, bone=rawbone, method=method)
    return angle_handlers.degreeify(raw_func)

def asymmetry_angle(data):
    left_ankle = np.array(data['AnkleLeft'])
    right_ankle = np.array(data['AnkleRight'])
    left_hand = np.array(data['HandLeft'])
    right_hand = np.array(data['HandRight'])
    left_hip = np.array(data['HipLeft'])
    right_hip = np.array(data['HipRight'])

    try:   # use spine as proxy to fix instability in feet
        asym_vec = 0.5 * (left_hand.T[0:2] + right_hand.T[0:2]).T - np.array(data['HipCenter']).T[0:2].T

    except KeyError:   # kludge: take advantage of fact that qualysis has no spine to get NIOSH ground truth
        asym_vec = 0.5 * (left_hand.T[0:2] + right_hand.T[0:2] -
                         left_ankle.T[0:2] - right_ankle.T[0:2]).T

    dist = 0.5 * (left_hand.T[0:2] + right_hand.T[0:2] - left_hip.T[0:2] - right_hip.T[0:2]).T

    try:
        # assuming thorax quat has x axis pointing in sagittal plane, as per Kevin email
        thorax = np.array(data[data_model.kin_bone_name(('HipCenter','HipCenter'))])
        a = thorax.T[0]
        b = thorax.T[1]
        c = thorax.T[2]
        d = thorax.T[3]
        i = a * a + b * b - c * c - d * d
        j = 2.0 * (a * c - b * d)
        #k = 2.0*(b*d) Keep this here for completeness, don't need it for projection
        sag_vec = -1.0 * np.vstack((i,j)).T
    except: # kinect now has no quaternions, but hip came from hipcenterq joints anyway
        a = np.array(data['HipRight']) - np.array(data['HipCenter'])
        b = np.array(data['HipLeft']) - np.array(data['HipCenter'])
        c = np.cross(a, b)
        sag_vec = c.T[0:2].T
    multiplier = np.sqrt((dist**2).sum(-1)) > 0.15
    mult = np.vectorize(lambda x: 1.0 if x == 1 else np.nan)
    raw_func = np.sum(normalize_rows(asym_vec) * normalize_rows(sag_vec), axis=-1) * mult(multiplier)
    return angle_handlers.degreeify(raw_func)

FUNCTIONS.append(Func(asymmetry_angle, "Asymmetry Angle", limits=[0,180]))        

for ang in data_model.ANGLES:
    if not "Ankle" in ang[1]:
        FUNCTIONS.append(Func(partial(quat_joint_angle, ang[1]), ang[1], limits=[0,180]))

HANDLEFT = 'WristLeft'
HANDRIGHT = 'WristRight'


# HIPCENTER IS PROBABLY BETTER THAN FEET FOR THIS
# FOR HORIZONTAL DISPLACEMENT.
# JUST TRIED IT, AND IT KINDA LOOKS WORSE....
def horiz_loc(data):
    left_hand = np.array(data[HANDLEFT])
    right_hand = np.array(data[HANDRIGHT])
    left_foot = np.array(data['FootLeft'])
    right_foot = np.array(data['FootRight'])
    try:
       # cheese = data['fail']
        raw_vec = (0.5 * (left_hand.T[0:2] + right_hand.T[0:2]) - \
                          np.array(data['HipCenter']).T[0:2]).T
    except KeyError: # kludge to use traditional measure with qual data: no spine marker on qual
        raw_vec = (0.5 * (left_hand.T[0:2] + right_hand.T[0:2] -
                         left_foot.T[0:2] - right_foot.T[0:2])).T
    return np.sqrt(raw_vec**2).sum(-1)

FUNCTIONS.append(Func(horiz_loc, "Horizontal Location", limits=[0.0,2.0]))

# LOOKS STUPIDLY WRONG: FEET DON'T ALWAYS INDICATE GROUND, AND THIS
# SHOULD REALLY BE HEIGHT ABOVE GROUND!!!
# FOR OUR TESTS, WE HAD THE SAME KINECT GROUND HEIGHT ALL THE TIME
# IN GENERAL CASE, NEED TO PASS A MESSAGE TO SET GROUND, USING EITHER CLIPPING PLANE
# OR SOME KIND OF SUITABLE AVERAGE OF FEET? OR EMPIRICAL DECISION
# BENEFIT OF FEET: IF WORKER IS ON A PLATFORM
# I WONDER IF FEET HEIGHT MEAN WORKS?
# JUST TRIED BELOW, LOOKS BETTER
def vert_loc(data, floor=0):
    left_hand = np.array(data[HANDLEFT])
    right_hand = np.array(data[HANDRIGHT])
    #left_foot = np.array(data['FootLeft'])
    #right_foot = np.array(data['FootRight'])
    raw_vec = 0.5 * (left_hand.T[2] + right_hand.T[2] -
                     floor)#left_foot.T[2] - right_foot.T[2])

    return raw_vec

FUNCTIONS.append(Func(vert_loc, "Vertical Location", limits=[0.0,2.0]))

def limb_length(limb, data):
    limb = np.array(data[limb[1]]) - np.array(data[limb[0]])
    return np.sqrt((limb**2).sum(-1))

def limb_x(limb, data):
    limb = np.array(data[limb[1]]) - np.array(data[limb[0]])
    return limb.T[0]
    
def limb_y(limb, data):
    limb = np.array(data[limb[1]]) - np.array(data[limb[0]])
    return limb.T[1]
    
def limb_z(limb, data):
    limb = np.array(data[limb[1]]) - np.array(data[limb[0]])
    return limb.T[2]

def coord(key, i, data):
    return np.array(data[key]).T[i]
    
for segment in data_model.KIN_SEGMENTS:
    FUNCTIONS.append(Func(partial(limb_length, data_model.KIN_SEGMENTS[segment]), 
                          segment,
                          limits=[0.1,0.7]))
    COORD_FUNCTIONS.append(Func(partial(limb_x, data_model.KIN_SEGMENTS[segment]),
                                segment+"x"))
    COORD_FUNCTIONS.append(Func(partial(limb_y, data_model.KIN_SEGMENTS[segment]),
                                segment+"y"))
    COORD_FUNCTIONS.append(Func(partial(limb_z, data_model.KIN_SEGMENTS[segment]),
                                segment+"z"))

for key in data_model.MAPPING:
    if data_model.MAPPING[key] is not None:
        for i in range(3):
            COORD_FUNCTIONS.append(Func(partial(coord, key, i), key + str(i)))


def make_data_sets(names=['master'],
                   normalized=False):
    # Will have to clean out nans on import into any particular modeler
    directory = TRAINING_DIR
    labels = [label for label in data_model.KIN_LABELS]
    newlabels = [label for label in labels]
    ending = "normalized" if normalized else ""
    if not os.path.exists(directory):
        os.makedirs(directory)
    end_X = None    
    for name in names:        
        with open("DataFiles/master/" + name + ".json", "r") as file:
            j = json.load(file)
            if normalized:
                raw = normalize(j, labels)                
                newlabels.remove('HipCenter')
            else:
                raw = j['kin']
            X = np.column_stack((raw[key] for key in newlabels)) #WHICH LABELS/FEATURES TO USE? INCL QUATS?? (AS HERE)
            if end_X == None: 
                end_X = X
            else:
                end_X = np.vstack((end_X, X))
    joblib.dump(end_X, directory + "X" + ending, compress=3)
    for function in FUNCTIONS:
        end_y = None
        for name in names:        
            with open("DataFiles/master/" + name + ".json", "r") as file:
                j = json.load(file)
                y = function.func(j['qual']) - function.func(j['kin'])
                if end_y == None:
                    end_y = y
                else:
                    end_y = np.hstack((end_y, y))
                joblib.dump(end_y, directory + "y" + function.name.replace(" ",""), compress=3)
    for function in COORD_FUNCTIONS:
        end_y = None
        for name in names:        
            with open("DataFiles/master/" + name + ".json", "r") as file:
                j = json.load(file)
                y = function.func(j['qual']) - function.func(j['kin'])
                if end_y == None:
                    end_y = y
                else:
                    end_y = np.hstack((end_y, y))
                joblib.dump(end_y, directory + "y" + function.name.replace(" ",""), compress=3)
    


class Inspector(object):
    """Class to aid in visualization of a function. Maybe belongs in separate file...?"""
    def __init__(self, function=None, name=None, labels=data_model.KIN_LABELS):
        self.func = function
        self.labels = []
        for label in labels:
            if type(label) == tuple: 
                label = data_model.kin_bone_name(label)
            self.labels.append(label)
        if name is None: name  = 'master'
        if function is None: 
            raise ValueError('Function to model needs to be passed in as "function" parameter')
        with open("DataFiles/master/" + name + ".json", "r") as file:
            j = json.load(file)
            X = np.column_stack((j['kin'][key] for key in self.labels))
            k = function(j['kin'])
            q = function(j['qual'])
            y = function(j['qual']) - function(j['kin'])
            self.X = X[~np.isnan(y), :]
            self.y = y[~np.isnan(y)]
            self.k = k[~np.isnan(y)]
            self.q = q[~np.isnan(y)]

    def scatter(self):
        pylab.figure()
        pylab.scatter(self.k, self.q)


def normalize(j, labels):
    temp = {label: np.array(j['kin'][label]) for label in labels}
    N = temp['HipCenter']
    out = {}
    for label in labels:
        out[label] = None
        if label != "HipCenter" and np.shape(temp[label])[1] == 3:
            out[label] = temp[label] - N
        else:
            out[label] = temp[label]
    return out
    
    
class Corrector(object):
    """Generic corrector class. Specify predictor in each subclass"""
    def __init__(self, 
                 function=None, 
                 normalized=False):
        self.predictor = None # replace in each subclass!
        self.func = function
        if function is None: 
            raise ValueError('Function to model needs to be passed in as "function" parameter')
        ending = "normalized" if normalized else ""
        raw_X = joblib.load(TRAINING_DIR + "X" + ending)
        raw_y = joblib.load(TRAINING_DIR + "y" + function.name.replace(" ",""))
        self.X = raw_X[~np.isnan(raw_y)]
        self.y = raw_y[~np.isnan(raw_y)]


    def train(self):
        self.predictor.fit(self.X,self.y)

    def split_train(self, state=0):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(self.X, range(len(self.y)), 
                                                                             test_size=0.4, random_state=state)
        self.test_indices = y_test
        self.predictor.fit(X_train, [self.y[i] for i in y_train])
        self.X_test = X_test
        self.y_test = [self.y[i] for i in y_test]

    def predict(self, dic):
        return self.predictor.predict(np.column_stack((dic[label] for label in self.labels)))

    def predict_vec(self, vec):
        return self.predictor.predict(vec)

class SVCorrector(Corrector):
    """support vector regressor"""
    def __init__(self, 
                 function=None, 
                 normalized=False):
        super(SVCorrector, self).__init__(function=function, 
                                          normalized=normalized) # get this right; not quite        
        self.predictor = SVR(kernel='poly')


class Treerector(Corrector):
    """Decision forest regressor"""
    def __init__(self, 
                 function=None, 
                 normalized=False,
                 n_trees=50,
                 max_depth=None):
        super(Treerector, self).__init__(function=function,  
                                         normalized=normalized)
        self.predictor = RandomForestRegressor(n_estimators=n_trees,
                                               n_jobs=-1, 
                                               max_depth=max_depth, 
                                               oob_score=True)


class GBTreerector(Corrector):
    """Decision forest regressor"""
    def __init__(self, 
                 function=None, 
                 normalized=False,
                 n_trees=5000,
                 max_depth=None):
        super(GBTreerector, self).__init__(function=function,  
                                           normalized=normalized)
        self.predictor = GradientBoostingRegressor(subsample=0.1, 
                                                   n_estimators=n_trees,
                                                   max_depth=max_depth)


def build_correctors(split=False, load=True):
    predictors = dict()
    all = "DFR/predictors.pkl"
    try:
        predictors = joblib.load(all)
        print "regressors loaded"
        if not load: # weird thing to do. hm.
            del(predictors)
            predictors = dict()
    except:
        for func in FUNCTIONS:
            file_name = "DFR/" + func.name + ".pkl"
            try:
                output = joblib.load(file_name)
                #with open(file_name, "rb") as file:
                #    if load:
                #        predictors[func.name] = pickle.load(file)
                if load:
                    predictors[func.name] = output.predictor
                    print "regressor %s loaded" % func.name
                else:
                    del(output)
            except:
                output = Treerector(function=func.func)
                if not split:
                    output.train()
                else:
                    output.split_train(state=random.randint(0,100))
                joblib.dump(output.predictor, file_name, 9)
                #with open("DFR/" + func.name + ".pkl", "wb") as file:
                #    joblib.dump(predictors[func.name], file, 9)
                print "regressor %s created and saved" % func.name
                if load: 
                    predictors[func.name] = output.predictor
                else:
                    del(output)
            if load: 
                try:
                    joblib.dump(predictors, "DFR/predictors.pkl", 9)
                except:
                    pass
    return predictors

def build_correctors_pickle(names=['master'], 
                            split=False, 
                            load=True, 
                            dump=False,
                            symmetrized=False):
    predictors = dict()
    #all = "DFR/predictors.pkl"
    terminator = "symm.pkl" if symmetrized else ".pkl"
    try:
        with open("DFR/predictors" + terminator, 'rb') as file:
            predictors = pickle.load(file)
            print "regressors loaded"
            if not load: # weird thing to do. hm.
                del(predictors)
                predictors = dict()
    except:
        grow_correctors(names=names, split=split, symmetrized=symmetrized)
    return predictors

def reduce_jobs(predictor_dict):
    for pred in predictor_dict:
        predictor_dict[pred].set_params(n_jobs=1)

correctors = {"DFR": Treerector, 
              "GBR": GBTreerector,
              "SVR": SVCorrector}


def grow_correctors(names=['master'], 
                    split=False, 
                    normalized=False,
                    keep_existing=True,
                    symmetrized=False,
                    n_trees=None,
                    max_depth=None,
                    kind="DFR"):
    predictors = {}
    if n_trees is None:
        n_trees = 250 if kind == "DFR" else 5000
    directory = kind + "N/" if normalized else kind + "/"
    directory += "{0}-{1}/".format(str(n_trees), str(max_depth))
    terminator = "symm.pkl" if symmetrized else ".pkl"
    newnames = []
    for name in names:
        newnames.append(name)
        if symmetrized:
            newnames.append(name + "symm")
    if keep_existing:
        try:        
            with open(directory + "predictors" + terminator, "rb") as file:
                predictors = pickle.load(file)
        except:
            print "No existing regressor aggregate of this kind, making new ones"
    jobs = []
    # train each model in a separate process (what happens with n_jobs for DFR?)
    for i in range(len(FUNCTIONS)):
        func = FUNCTIONS[i]        
        if func.name in predictors: continue
        j = multiprocessing.Process(target=grow_corrector, 
                        kwargs={'directory': directory,
                                'function': func,
                                'terminator': terminator,
                                'names': newnames,
                                'normalized': normalized,
                                'split': split,
                                'kind': kind,
                                'n_trees': n_trees,
                                'max_depth': max_depth})
        jobs.append(j)
        j.start()
        if kind == "DFR": j.join()
    # wait for each job to finish
    if kind == "GBR":
        for j in jobs:
            j.join()
    # collect all of the predictors and save them to a master file
    for func in FUNCTIONS:
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + func.name.replace(" ","") + terminator, "rb") as file:
            predictors[func.name] = pickle.load(file)
    with open(directory + "predictors" + terminator, 'wb') as file:
        pickle.dump(predictors, file, -1)


def grow_corrector(directory=None,
                   function=None, 
                   terminator=None,
                   names=None, 
                   normalized=False, 
                   split=False,
                   kind="DFR",
                   n_trees=None,
                   max_depth=None):
    if n_trees is None:
        n_trees = 50 if kind == "DFR" else 5000        
    file_name = directory + function.name.replace(" ","") + terminator
    output = correctors[kind](names=names, 
                              function=function.func,
                              normalized=normalized,
                              n_trees=n_trees,
                              max_depth=max_depth)
    if not split:
        output.train()
    else:
        output.split_train(state=random.randint(0,100))
    try:
        output.predictor.set_params(n_jobs=1)
    except ValueError: # only necessary for DFR
        pass
    
    if not os.path.exists(directory):
        os.makedirs(directory)    
    with open(file_name, "wb") as file:
        pickle.dump(output.predictor, file, -1)
        print "regressor %s created and saved" % function.name    


def grow_svr_correctors(name='master', split=False, normalized=False):
    predictors = {}
    directory = "SVRN/" if normalized else "SVR/"
    for func in FUNCTIONS:
        file_name = directory + func.name + ".pkl"
        output = SVCorrector(name=name, 
                                      function=func.func,
                                      normalized=normalized)
        if not split:
            output.train()
        else:
            output.split_train(state=random.randint(0,100))
        try:
            output.predictor.set_params(n_jobs=1)
        except ValueError: #only works if using a parallelizable thing like DFR
            pass
        with open(file_name, "wb") as file:
            pickle.dump(output.predictor, file, -1)
            print "regressor %s created and saved" % func.name
            predictors[func.name] = output.predictor
    with open(directory + "predictors.pkl", 'wb') as file:
        pickle.dump(predictors, file, -1)


# support vector regressors
def build_corrector_model(name='master'):
    try:
        with open("predictors.pkl", "rb") as file:
            predictors = pickle.load(file)
    except:
        predictors = dict()
        for func in FUNCTIONS:
            predictors[func.name] = Corrector(name, func.func)
        with open("predictors.pkl", "wb") as file:
            pickle.dump(predictors, file, -1)
    return predictors
       

def plot_correctors(predictors=None, 
                    name='master',
                    json_data=None, 
                    functions=None,
                    alpha=0.1,
                    figsize=None,
                    normalized=False,
                    ending=None):
    labels = data_model.KIN_LABELS
    newlabels = []
    directory = "DFRN/" if normalized else "DFR/"
    if ending is None:
        ending = "N.png" if normalized else ".png"
    for label in labels:
        if type(label) == tuple:
            label = data_model.kin_bone_name(label)
        newlabels.append(label)
    if predictors is None:
        with open(directory + "predictors.pkl", "rb") as pickles:
            predictors = pickle.load(pickles)
    if functions is None:
        functions = [func.name for func in FUNCTIONS]
    with open("DataFiles/master/" + name + ".json", "rb") as file:
        if json_data is None:
            j = json.load(file)
        else:
            j = json_data
        if normalized:
            d = normalize(j, newlabels)
            newlabels.remove('HipCenter')
        else:
            d = j['kin']
        for func in FUNCTIONS:
            a = func.limits[0]
            b = func.limits[1]
            if func.name in functions:
                corr = predictors[func.name]
                rk = func.func(j['kin'])
                rq = func.func(j['qual'])
                mk = corr.predict(np.column_stack((d[key] for key in newlabels))) + rk
                pylab.figure(func.name, figsize=figsize)
                pylab.subplot(121, aspect='equal')
                pylab.scatter(rk, rq, alpha=alpha)
                pylab.axis([a,b,a,b])
                pylab.subplot(122, aspect='equal')
                pylab.scatter(mk, rq, alpha=alpha)
                pylab.axis([a,b,a,b])
                pylab.savefig("Plots/" + func.name + ending, bbox_inches=0)
#        pylab.show()
        

def test_correctors(predictors=None, 
                    name='master',
                    json_data=None, 
                    functions=None,
                    normalized=False,
                    ending=None,
                    kind="DFR",
                    ns=None):
    newlabels = data_model.KIN_LABELS
    if predictors is None:
        predictors = {}
        if ns is None:
            if kind == "GBR":
                ns = [1000, 5000, 10000]
            if kind == "DFR":
                ns = [100, 1000, 5000]
        for n in ns:
            with open(kind + "/" + str(n) + "/predictors.pkl", "rb") as pickles:
                predictors[n] = pickle.load(pickles)
    if functions is None:
        functions = [func.name for func in FUNCTIONS]
    err = {}
    with open("DataFiles/master/" + name + ".json", "rb") as file:
        if json_data is None:
            j = json.load(file)
        else:
            j = json_data
        d = j['kin']
        for func in FUNCTIONS:
            err[func.name] = []
            if func.name in functions:
                for n in ns:
                    corr = predictors[n][func.name]
                    rk = func.func(j['kin'])
                    rq = func.func(j['qual'])
                    mk = corr.predict(np.column_stack((d[key] for key in newlabels))) + rk
                    gap = mk - rq
                    g = gap[~np.isnan(gap)]
                    err[func.name].append(np.sqrt((g**2).sum() / len(rq)))
                print func.name
                print "%f, %f, %f" % (err[func.name][0], err[func.name][1], err[func.name][2])
                pylab.figure(func.name)
                pylab.plot(ns, err[func.name])

def plot_correctors_bin(predictors=None, 
                    name='master',
                    json_data=None, 
                    functions=None,
                    figsize=None):
    labels = data_model.KIN_LABELS
    newlabels = []
    for label in labels:
        if type(label) == tuple:
            label = data_model.kin_bone_name(label)
        newlabels.append(label)
    if predictors is None:
        with open("DFR/predictors.pkl", "rb") as pickles:
            predictors = pickle.load(pickles)
    if functions is None:
        functions = [func.name for func in FUNCTIONS]
    with open("DataFiles/master/" + name + ".json", "rb") as file:
        if json_data is None:
            j = json.load(file)
        else:
            j = json_data
        for func in FUNCTIONS:
            a = func.limits[0]
            b = func.limits[1]
            if func.name in functions:
                corr = predictors[func.name]
                rk = func.func(j['kin'])
                rq = func.func(j['qual'])
                mk = corr.predict(np.column_stack((j['kin'][key] for key in newlabels))) + rk
                pylab.figure(func.name + ' bin', figsize=figsize)
                pylab.subplot(121, aspect='equal')
                pylab.hexbin(rk, rq, cmap='binary')
                pylab.axis([a,b,a,b])
                pylab.subplot(122, aspect='equal')
                pylab.hexbin(mk, rq, cmap='binary')
                pylab.axis([a,b,a,b])


def plot_correctors_separate(predictors=None, 
                             name='master', 
                             json_data=None, 
                             functions=None,
                             alpha=1.0):
    labels = data_model.KIN_LABELS
    newlabels = []
    for label in labels:
        if type(label) == tuple:
            label = data_model.kin_bone_name(label)
        newlabels.append(label)
    if predictors is None:
        with open("DFR/predictors.pkl", "rb") as pickles:
            predictors = pickle.load(pickles)
    if functions is None:
        functions = [func.name for func in FUNCTIONS]
    with open("DataFiles/master/" + name + ".json", "rb") as file:
        if json_data is None:
            j = json.load(file)
        else:
            j = json_data
        for func in FUNCTIONS:
            if func.name in functions:
                corr = predictors[func.name]
                rk = func.func(j['kin'])
                rq = func.func(j['qual'])
                mk = corr.predict(np.column_stack((j['kin'][key] for key in newlabels))) + rk
                pylab.figure(func.name + " raw")
                pylab.scatter(rk, rq, alpha=alpha)
                pylab.figure(func.name + " mod")
                pylab.scatter(mk,rq, alpha=alpha)

if __name__ == '__main__':
    plot_correctors()