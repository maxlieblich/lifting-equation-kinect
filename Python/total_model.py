import data_model
import angle_handlers
import loc_handlers
import correlator
import os
import sys
import cPickle as pickle
import body_model

# some pickling stuff to get methods pickled.
# maybe make this a separate module and import it
# elsewhere, if needed?

SUBJECTS = 6
TRIALS = ['r', 'l', 'w', 'u']

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = None
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name is not None: 
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)




class TotalModel(object):

    def __init__(self, name, load=False, use_body_model=True):
        if load:
            try:
                with open("DataFiles/pickles/" + name + ".pkl", "rb") as file:
                    self = pickle.load(file)
            except:
                self.populate(name, use_body_model=use_body_model)
        else:
            self.populate(name, use_body_model=use_body_model)

    def populate(self, name, use_body_model=True):
        self.name = name
        kin_path = "DataFiles/kin/" + name + ".csv"
        qual_quat_path = "DataFiles/qual/" + name + "q.csv"
        qual_joint_path = "DataFiles/qual/" + name + "c.csv"
        self.DataModel = data_model.DataModel(kin_file_name=kin_path, 
                                              qual_file_name=qual_joint_path, 
                                              qual_quat_file_name=qual_quat_path)
        if use_body_model:
            self.BodyModel = body_model.BodyModel(self.DataModel.kin.data.raw)
            self.BodyModel.greedy_fit() # since BodyModel data points to DataModel.kin.data.raw, it should update that as well
        self.DataModel.smooth_data()
        self.AngleHandlers = angle_handlers.AngleHandlers(self.DataModel)
        self.QuatAngleHandlers = angle_handlers.QuatAngleHandlers(self.DataModel)
        self.HorizLocHandler = loc_handlers.HorizLocHandler(self.DataModel)
        self.VertLocHandler = loc_handlers.VertLocHandler(self.DataModel)
        # without labels on hipcenter, etc., cannot really correct all lengths
        # hm....
#        self.BodyModel = body_model.BodyModel(self.DataModel)
        self.WristsCorrelator = correlator.WristsCorrelator(self.DataModel)
        self.WristsCorrelator.set_shift()

    def repopulate(self, use_body_model=True):
        if use_body_model:
            self.BodyModel = body_model.BodyModel(self.DataModel.kin.data.raw)
            self.BodyModel.greedy_fit() # since BodyModel data points to DataModel.kin.data.raw, it should update that as well
        self.DataModel.smooth_data()
        self.AngleHandlers = angle_handlers.AngleHandlers(self.DataModel)
        self.QuatAngleHandlers = angle_handlers.QuatAngleHandlers(self.DataModel)
        self.HorizLocHandler = loc_handlers.HorizLocHandler(self.DataModel)
        self.VertLocHandler = loc_handlers.VertLocHandler(self.DataModel)
        # without labels on hipcenter, etc., cannot really correct all lengths
        # hm....
        #        self.BodyModel = body_model.BodyModel(self.DataModel)
        self.WristsCorrelator = correlator.WristsCorrelator(self.DataModel)
        self.WristsCorrelator.set_shift()

    def dump_data(self):
        json_data = self.DataModel.dump_data()
        with open("Web/json/"+self.name+".json", "wb") as file:
            file.write(json_data)

    def dump_data_sync(self, use_body_model=True):
        """Dump synchronized smoothed data"""
        import json
        
        start = self.DataModel.qual.data.raw['Time'][0]
        end = self.DataModel.qual.data.raw['Time'][-1]
        times = self.DataModel.kin.data.raw['Time']
        shift = self.DataModel.time_shift
        good_times_kin = times[(times >= start - shift) & (times <= end - shift)]
        good_times_qual = good_times_kin + shift
        # alert: the following will reprocess the whole kin data model
        # to constrain times to lie in the range determined by the original
        # sync. Any further use of the model will be working with this constrained model
        # IS THIS TOO DANGEROUS?
        self.DataModel.kin.constrain_times(good_times_kin)
        self.DataModel.qual.constrain_input_times(good_times_qual)
        self.DataModel.qual.smooth_data(s=0, times=good_times_qual)
        #self.repopulate(use_body_model=use_body_model)

        #now that everyone's lined up, feel free to compute affine
        #transformation between kinect and qualysis
        coef, intercept = self.DataModel.compute_affine()
        self.DataModel.align()
        
        json_data = self.DataModel.dump_data()
        with open("Web/json/" + self.name + ".json", "wb") as file:
            file.write(json_data)
            
        with open("Web/json/" + self.name + "coefs.json", "wb") as file:
            json.dump({"coef": coef.tolist(), "intercept": intercept.tolist()}, 
                       file)


def preserve(TM, name=None):
    if name is None: name = TM.name
    with open("DataFiles/pickles/" + name + ".pkl", "w") as file:
        pickle.dump(TM, file, -1)

def quat_check_corr(TM):
    k = TM.QuatAngleHandlers.kin.raw[('ElbowLeft','WristLeft')]
    kt = TM.DataModel.kin.data.raw['Time']
    q = TM.QuatAngleHandlers.qual.smooth[('ElbowLeft','WristLeft')]
    qt = TM.DataModel.qual.data.raw['Time']
    shift = TM.DataModel.time_shift
    k.plot(kt)
    q.plot(qt, shift)

def check_corr(TM):
    k = TM.AngleHandlers.kin.raw['ElbowLeft']
    kt = TM.DataModel.kin.data.raw['Time']
    q = TM.AngleHandlers.qual.smooth['ElbowLeft']
    qt = TM.DataModel.qual.data.raw['Time']
    shift = TM.DataModel.time_shift
    k.plot(kt)
    q.plot(qt, shift)




all_models = dict()

def load_all():
    with open("DataFiles/pickles/all.pkl", "rb") as file:
        return pickle.load(file)
    

def create_all(subjects = None, use_body_model=True):
    if subjects is None:
        subjects = range(1, SUBJECTS + 1)
    for i in subjects:
        for trial in TRIALS:
            name = str(i) + trial
            print name
            try:
                all_models[name] = TotalModel(name, use_body_model=use_body_model)
                all_models[name].dump_data_sync(use_body_model=use_body_model)
            except Exception:
                print "Error {0}: {1}".format(name, sys.exc_info()[0])
    
def save_all():
    with open("DataFiles/pickles/all.pkl", "wb") as file:
        pickle.dump(all_models, file, -1)
    

class AllModels(object):
    
    def __init__(self, load="False"):
        self.models = dict()
        if load is True:
            with open("DataFiles/pickles/all.pkl", "rb") as file:
                self = pickle.load(file)
        else:
            self.create()

    def create(self):
        daters = os.listdir("DataFiles/kin")
        for dater in daters:
            name = dater.split(".")[0]
            self.models[name] = TotalModel(name)

    def save(self):
        with open("DataFiles/pickles/all.pkl", "wb") as file:
            pickle.dump(self, file, -1)
    


if __name__ == '__main__':
    #create_all()
    #save_all()
    TM = TotalModel("4r")
    TM.dump_data_sync()
    quat_check_corr(TM)