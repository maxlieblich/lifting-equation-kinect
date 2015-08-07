import numpy as np
import pandas as pd
import json
import os
import random
from utilities import normalize_rows
import angle_handlers
import correctors
import data_model
import quaternions

TIME_CUTOFF = 0.4

#max_time - shift >= time >= -shift
#dirt(time + shift) < TIME_CUTOFF

def check_time(dirt, time, shift):
    if time + shift < 0: 
        return False
    else:
        try:
            return dirt(time + shift) < TIME_CUTOFF
        except ValueError:
            return False

def make_row(q, dirt, label, time, shift):
    output = []
    if check_time(dirt, time, shift):
        try:
            output = np.array(q[label](time + shift))
        except TypeError:
            output = np.array([q[label][i](time + shift) for i in range(3)])
    else:
        output = [np.nan, np.nan, np.nan]
    return output

class FlaggedPositionPool(object):
    """Eat one of the flagged files, hand cleaned with web interface. Uses pandas.DataFrame
    for easy merging"""
    def __init__(self, name=None, body_model=False, fit_limbs=False):
        print name
        self.kin = pd.DataFrame()
        self.qual = pd.DataFrame()
        if name is not None:
            with open ("Web/json/" + name + ".json", "r") as data_file:
                with open("Web/json/flagged/" + name + ".json", "r") as flag_file:
                    jdata = json.load(data_file)
                    jflag = json.load(flag_file)
                    self.kin = pd.DataFrame.from_dict(jdata["kin"])
                    flags = pd.Series(jflag["flags"])
                    if len(self.kin['Time']) > len(flags):
                        flags = flags.append(pd.Series(np.ones(len(self.kin['Time']) - len(flags)), index=range(len(flags), len(self.kin['Time']))))
                    self.kin = self.kin[flags == 0]
                    self.qual = pd.DataFrame.from_dict(jdata["qual_smooth"])
                    self.qual = self.qual[flags == 0]
                    if body_model:
                        import body_model
                        data_dict = self.kin.to_dict(outtype='list')
                        BM = body_model.BodyModel(data_dict)
                        if fit_limbs: BM.fit_limbs()
                        #BM.scale()
                        self.kin = pd.DataFrame.from_dict(BM.data)

    def generate_dict(self):
        return {"kin": self.kin.to_dict(outtype='list'), "qual": self.qual.to_dict(outtype='list')}
    #    self.kin = self.DF["kin"]
    #    self.qual = self.DF["qual_smooth"]


def flagged_merge(pools):
    """Eat list of pools, merge them"""
    final = None
    for pool in pools:
        if final is None:
            final = pool
        else:
            final.kin = final.kin.append(pool.kin, ignore_index=True)
            final.qual = final.qual.append(pool.qual, ignore_index=True)
    return final

def flagged_dump(names, master_name=None):
    pools = [FlaggedPositionPool(name) for name in names]
    if master_name is None:
        master_name = names.join()
    pool = flagged_merge(pools)
    with open("DataFiles/master/" + master_name + ".json", "w") as file:
        file.write(json.dumps(pool.generate_dict()))
    return pool


def master_flagged_dump(outname='master', excluded=[], types=None, body_model=False):
    """Eat list of pools, output json with collected data"""
    names = [item[0:-5] for item in os.listdir("Web/json/flagged")]
    for e in excluded:
        names = [name for name in names if not e in name] # stupid way to do this
    if types is not None:
        names = [name for name in names for t in types if t in name]
    pools = [FlaggedPositionPool(name=name, body_model=body_model) for name in names]
    pool = flagged_merge(pools)
    with open("DataFiles/master/" + outname + ".json", "w") as file:
        file.write(json.dumps({"kin": pool.kin.to_dict(outtype='list'), "qual": pool.qual.to_dict(outtype='list')}))
    return pool


def reflect(cols):
    cols = [name.replace("Left", "Heft") for name in cols]
    cols = [name.replace("Right", "Left") for name in cols]
    cols = [name.replace("Heft", "Right") for name in cols]
    return cols


def reflect_vect(vec):
    if type(vec) == list:
        if len(vec) == 3:
            return [vec[0], -1*vec[1], vec[2]]
        if len(vec) == 4:
            return [vec[0], -1*vec[1], vec[2], -1*vec[3]]
    else:
        return vec


def symmetrize(name):
    """ 
        Only for master files now.
    """
    with open("DataFiles/master/" + name + ".json") as file:
        j = json.load(file)
        kin = pd.DataFrame.from_dict(j['kin'])
        qual = pd.DataFrame.from_dict(j['qual'])
        for key in kin:
            kin[key] = kin[key].apply(reflect_vect)
        for key in qual:
            qual[key] = qual[key].apply(reflect_vect)
        kin.columns = reflect(kin.columns)
        qual.columns = reflect(qual.columns)
    with open("DataFiles/master/" + name + "symm.json", "wb") as file:
        file.write(json.dumps(
                    {"kin": kin.to_dict(outtype='list'), 
                     "qual": qual.to_dict(outtype='list')}
                     ))

def master_symmetrize():
    names = [item[0:-5] for item in os.listdir("DataFiles/master") if "symm" not in item]
    for name in names:
        symmetrize(name)
        

def rotate(name, radians):
    m = np.array([[np.cos(radians), -np.sin(radians), 0], 
                  [np.sin(radians), np.cos(radians), 0], 
                  [0, 0, 1]])
    with open("DataFiles/master/" + name + ".json") as file:
        j = json.load(file)
        kin = pd.DataFrame.from_dict(j['kin'])
        qual = pd.DataFrame.from_dict(j['qual'])
        for key in kin:
            kin[key] = kin[key].apply(lambda x: m.dot(np.array(x)).tolist())
    with open("DataFiles/master/" + 
              name + str(radians).replace(".","-") + 
              "rot.json", "wb") as file:
        file.write(json.dumps(
                    {"kin": kin.to_dict(outtype='list'), 
                     "qual": qual.to_dict(outtype='list')}
                     ))

def rand_rot_mat():
    radians = np.pi * (random.random() - 0.5)
    return np.array([[np.cos(radians), -np.sin(radians), 0], 
                  [np.sin(radians), np.cos(radians), 0], 
                  [0, 0, 1]])

def rot_mat(radians):
    return np.array([[np.cos(radians), -np.sin(radians), 0], 
                  [np.sin(radians), np.cos(radians), 0], 
                  [0, 0, 1]])


def rand_rotate(name):
    with open("DataFiles/master/" + name + ".json") as file:
        j = json.load(file)
        kin = pd.DataFrame.from_dict(j['kin'])
        qual = pd.DataFrame.from_dict(j['qual'])
        for key in kin:
            kin[key] = kin[key].apply(lambda x: rand_rot_mat().dot(np.array(x)).tolist())
    with open("DataFiles/master/" + 
              name + "randrot.json", "wb") as file:
        file.write(json.dumps(
                    {"kin": kin.to_dict(outtype='list'), 
                     "qual": qual.to_dict(outtype='list')}
                     ))
    

def compile(name='master'):
    names = [item[0:-5] for item in os.listdir("DataFiles/master") if name in item and item[-5:] == ".json"]
    final = {"kin": None, "qual": None}
    for name in names:
        with open("DataFiles/master/" + name + ".json","rb") as file:
            j = json.load(file)
            kin = pd.DataFrame.from_dict(j['kin'])
            qual = pd.DataFrame.from_dict(j['qual'])
            if final['kin'] is None:
                final['kin'] = kin
                final['qual'] = qual
            else:
                final['kin'] = final['kin'].append(kin, ignore_index=True)
                final['qual'] = final['qual'].append(qual, ignore_index=True)
    with open("DataFiles/master/" + name + "compiled.json","wb") as file:
        file.write(json.dumps(
                              {"kin":final['kin'].to_dict(outtype='list'),
                               "qual":final['qual'].to_dict(outtype='list')}
                    ))


HINGES = ["ElbowRight", "ElbowLeft", "KneeRight", "KneeLeft"]
BALLS = ["ShoulderRight", "ShoulderLeft", "HipRight", "HipLeft"] # feet and wrists too

class LengthAndAngleRepresentation(object):
    # keep representation of torso with coords for now,
    # focus on arms and legs.
    # Main idea: turn angles into coordinates in hope that
    # this will make linear/quadratic regression more successful
    def __init__(self, data=None):
        if data is None:
            with open("DataFiles/master/master.json", "rb") as file:
                data = json.load(file)
        self.coords = data
        try:
            self.kin = data['kin']
            self.qual = data['qual']
        except KeyError:
            self.kin = data
        for hinge in HINGES:
            self.kin[hinge + 'Angle'] = angle_handlers.z_angle(data=self.kin,
                                                             bone=[hinge],
                                                             method='kin')

            normal_fore_limb = None
            normal_upper_limb = None
            ball_bone = None
            for seg in data_model.KIN_SEGMENTS:
                bone = data_model.KIN_SEGMENTS[seg]
                add = None
                if hinge == bone[0]:
                    add = 'PostLength'
                    normal_fore_limb = normalize_rows(np.array(self.kin[bone[1]]) - np.array(self.kin[bone[0]]))
                if hinge == bone[1]:
                    add = 'PreLength'
                    normal_upper_limb = normalize_rows(np.array(self.kin[bone[1]]) - np.array(self.kin[bone[0]]))
                    ball_bone = bone[0]
                if add is not None:
                    self.kin[hinge + add] = correctors.limb_length(limb=bone, data=self.kin)

            z = normal_upper_limb
            x = normalize_rows(normal_fore_limb - np.dot((normal_fore_limb * normal_upper_limb).sum(-1).T, normal_upper_limb))
            y = np.cross(z,x)
            self.kin[ball_bone + "Axis"] = z
            self.kin[ball_bone + "Normal"] = y
        newlabels = []
        for label in data_model.KIN_LABELS:
            if label not in HINGES and label not in BALLS:
                newlabels.append(label)
            elif label in HINGES:
                newlabels.append(label+'Angle')
                newlabels.append(label+'PostLength')
                newlabels.append(label+'PreLength')
            elif label in BALLS:
                newlabels.append(label+'Axis')
                newlabels.append(label+'Normal')
        self.newkin = {label: self.kin[label] for label in newlabels}


    # def dump(self):
    #     joblib.dump("DataFiles/training/")


def proj(a,b):
    c = normalize_rows(b)
    return np.dot((a * c).sum(-1), c)

def center_of_mass(data):
    return 1.0/6.0 * (np.array(data['ShoulderLeft']) + np.array(data['ShoulderRight']) + 
           np.array(data['HipLeft']) + np.array(data['HipRight']) + 
           np.array(data['Spine']) + np.array(data['HipCenter']))

def normalize_data(data, center_floor = False):
    """Assume that data is aligned (so qual and kin have best fit)"""

    qual_present = hasattr(data, 'qual')

    k = data['kin']
    kn = dict()

    if qual_present:
        q = data['qual']
        qn = dict()

    center = center_of_mass(k)
    if center_floor:
        center *= np.array((1,1,0))

    # first translate center of mass to the origin
    for key in k.keys():
        if key in data_model.KIN_LABELS:
            kn[key] = np.array(k[key]) - center
        else:
            kn[key] = k[key]
    if qual_present:
        for key in q.keys():
            if len(q[key][0]) == 3:
                qn[key] = np.array(q[key]) - center
            else:
                qn[key] = q[key]

    # coordinate system to use: z axis, hip orientation vector (suitably
    # gram schmidted) --  call this x, whatever is necessary to make right-handed system -- y

    hr = kn['HipRight']
    hl = kn['HipLeft']
    n = normalize_rows(np.cross(hl, hr))
    x = normalize_rows(n - n * np.array([0,0,1]))
    y = normalize_rows(np.cross(np.array([0,0,1]), x))

    outk = {}
    if qual_present:
        outq = {}

    for key in kn.keys():
        if key in data_model.KIN_LABELS:
            outk[key] = np.column_stack(((kn[key] * x).sum(-1), (kn[key] * y).sum(-1), kn[key].T[2]))
        else:
            outk[key] = k[key]

    if qual_present:
        for key in qn.keys():
            if len(q[key][0]) == 3:
                outq[key] = np.column_stack(((qn[key] * x).sum(-1), (qn[key] * y).sum(-1), qn[key].T[2]))
            elif key != 'Time' and key != 'HipCenterQ':
                outq[key] = q[key]
            elif key == 'HipCenterQ':
                theta = np.arccos(x.T[0])
                s = np.cos(0.5 * theta)
                t = np.sin(0.5 * theta)
                outq[key] = quaternions.conjugate(np.column_stack([s, np.zeros(len(s)), np.zeros(len(s)), t]), np.array(q[key]))

    # try:
    #     outk['Time'] = k['Time']
    # except KeyError:
    #     pass
    if qual_present:
        try:
            outq['Time'] = q['Time']
        except KeyError:
            pass

    if qual_present:
        return {'unnormalized_kin': k, 'unnormalized_qual': q, 'kin': outk, 'qual': outq}
    else:
        return {'unnormalized_kin': k, 'kin': outk}


if __name__ == '__main__':
  #  master_flagged_dump()
    rand_rotate("master")
    symmetrize("masterrandrot")
