import numpy as np
from utilities import normalize_rows

CUTOFF = 0.001

def join_parts(x):
    if len(np.array(x).shape) == 1:
        return np.array(x)
    else:
        return np.column_stack(x)

def ensure_nbhd(q0, q1):
    return q1*np.clip((q0*q1).sum(-1), -1, 1)
    

def slerp(q0, q1, t):
    q1 = ensure_nbhd(q0, q1)
    angles = np.arccos((q0*q1).sum(-1))
    q = 1.0 / np.sin(angles) * q0 * np.sin((1.0 - t) * angles) + q1 * np.sin(t * angles)
    if len(q.shape) == 1:
        if np.isnan(q).any():
            q = q0
    else:
        for i in range(0, len(q)):
            if np.isnan(q[i]).any():
                q[i] = q0[i]
    return q
    
def prod(a, b):
    x = a.T[0] * b.T[0] - a.T[1] * b.T[1] - a.T[2] * b.T[2] - a.T[3] * b.T[3]
    y = a.T[0] * b.T[1] + a.T[1] * b.T[0] + a.T[2] * b.T[3] - a.T[3] * b.T[2]
    z = a.T[0] * b.T[2] + a.T[2] * b.T[0] + a.T[3] * b.T[1] - a.T[1] * b.T[3]
    w = a.T[0] * b.T[3] + a.T[3] * b.T[0] + a.T[1] * b.T[2] - a.T[2] * b.T[1]
    return join_parts((x,y,z,w))

def invert(a):
    r = np.sqrt((a*a).sum(-1))
    b = ((a.T) / r).T
    c = b * np.array([1, -1, -1, -1])
    return c


def rotation_between(a, b):
    # assuming both are normal
    modb = ensure_nbhd(a, b)
    return prod(invert(a), modb)


def quaternion_angle(a):
    b = normalize_rows(a)
    return 2.0 * np.arccos((b.T)[2].T)


def conjugate(q, a):
    return prod(prod(q, a), invert(q))

def act(q, a):  # now a is a three-dim vec
    if len(a.shape) == 1:
        obs = 1
        return conjugate(q, join_parts((0, a.T[0], a.T[1], a.T[2]))).T[1:].T

    else:
        obs = len(a)
        return conjugate(q, join_parts((np.zeros(obs), a.T[0], a.T[1], a.T[2]))).T[1:].T

def axis_mover(z):
    u = normalize_rows(np.column_stack((-1.0 * z.T[1], -1.0 * z.T[0])))
    alpha = np.arccos(z.T[2])
    if len(z.shape) == 1:
        x = np.cos(0.5 * alpha)
        w = 0.0
    else:
        x = np.cos(0.5 * alpha) * np.ones((z.shape[0],))
        w = np.zeros((z.shape[0],))
    return join_parts((x, np.sin(0.5 * alpha) * u.T[0], np.sin(0.5 * alpha) * u.T[1], w))


def canonical(q):
    z = conjugate(q, np.array([0.0, 0.0, 0.0, 1.0])).T[1:].T
    z = normalize_rows(z)
    return axis_mover(z)


def residual(q):
    return prod(invert(canonical(q)), q)


def residual_modulus(q):
    return residual(q).T[1] / (2 * np.pi)


    
class QuatSmoother(object):
    def __init__(self, 
                 max_deviation_radius=0.0, 
                 smoothing=0.25, 
                 correction=0.25, 
                 prediction=0.0, 
                 jitter_radius=0.0):
        # look like MS defaults, except for prediction
        self.max_deviation_radius=max_deviation_radius 
        self.smoothing=smoothing 
        self.correction=correction 
        self.prediction=prediction 
        self.jitter_radius=jitter_radius        
    
    def smooth(self, q):
        """Implement double exponential smoothing on generated timeseries."""
        """Copying most of MS code. Probably painfully slow in Python because
        it is not vectorized."""
        filt = np.zeros(q.shape)
        trend = np.zeros(q.shape)
        
        filt[0] = q[0]
        trend[0] = np.array([1.0, 0.0, 0.0, 0.0])
        
        filt[1] = slerp(q[0], filt[0], 0.5)
        diff_started = rotation_between(filt[1], filt[0])
        trend[1] = slerp(trend[0], diff_started, self.correction)
            
        for i in range(2, len(q)):
            diff_jitter = rotation_between(q[i], filt[i-1])
            diff_val_jitter = np.abs(quaternion_angle(diff_jitter))
            if diff_val_jitter <= self.jitter_radius:
                filt[i] = slerp(filt[i-1], q[i], diff_val_jitter/self.jitter_radius)
            else:
                filt[i] = q[i]
            filt[i] = normalize_rows(slerp(filt[i], prod(filt[i-1], trend[i-1]), self.smoothing))
            diff_jitter = rotation_between(filt[i], filt[i-1])
            trend[i] = normalize_rows(slerp(trend[i-1], diff_jitter, self.correction))
        
        return np.array(filt)


def circ(a):
    return np.fmod(a,1.0)

class CircleSmoother(object):

    def __init__(self, 
                 smoothing=0.25, 
                 correction=0.25,
                 jitter_radius=0.0,
                 prediction=0.0):
        # look like MS defaults, except for prediction
        self.smoothing=smoothing 
        self.correction=correction 
        self.prediction=prediction
        self.jitter_radius = jitter_radius
    
    def smooth(self, q):
        """Implement double exponential smoothing on generated timeseries."""
        """Probably painfully slow in Python because
        it is not vectorized."""
        filt = np.zeros(q.shape)
        trend = np.zeros(q.shape)
        
        filt[0] = q[0]
        trend[0] = 0.0
        
        filt[1] = circ(0.5*(q[0] + filt[0]))
        diff_vec = circ(filt[1] - filt[0])
        trend[1] = circ(self.correction * diff_vec + (1.0 - self.correction) * trend[0])
            
        for i in range(2, len(q)):
            diff_jitter = circ(q[i] - filt[i-1])
            diff_val_jitter = circ(np.sqrt((diff_jitter*diff_jitter).sum()))
            if diff_val_jitter <= self.jitter_radius:
                filt[i] = circ(q[i] * diff_val_jitter/self.jitter_radius + filt[i-1] * (1.0 - diff_val_jitter/self.jitter_radius))
            else:
                filt[i] = q[i]
            filt[i] = circ(filt[i] * (1.0 - self.smoothing) + (filt[i-1] + trend[i-1]) * self.smoothing)
            diff_jitter = circ(filt[i] - filt[i-1])
            trend[i] = circ(trend[i-1] * (1.0 - self.correction) + diff_jitter * self.correction)
        
        return np.array(filt)


                
class PositionSmoother(object):
    def __init__(self, 
                 max_deviation_radius=0.00, 
                 smoothing=0.25, 
                 correction=0.25, 
                 prediction=0.0, 
                 jitter_radius=0.00):
        # look like MS defaults, except for prediction
        self.max_deviation_radius=max_deviation_radius 
        self.smoothing=smoothing 
        self.correction=correction 
        self.prediction=prediction 
        self.jitter_radius=jitter_radius        
    
    def smooth(self, q):
        """Implement double exponential smoothing on generated timeseries."""
        """Probably painfully slow in Python because
        it is not vectorized."""
        filt = np.zeros(q.shape)
        trend = np.zeros(q.shape)
        
        filt[0] = q[0]
        trend[0] = np.array([0.0, 0.0, 0.0])
        
        filt[1] = 0.5*(q[0] + filt[0])
        diff_vec = filt[1] - filt[0]
        trend[1] = self.correction * diff_vec + (1.0 - self.correction) * trend[0]
            
        for i in range(2, len(q)):
            diff_jitter = q[i] - filt[i-1]
            diff_val_jitter = np.sqrt((diff_jitter*diff_jitter).sum())
            if diff_val_jitter <= self.jitter_radius:
                filt[i] =q[i] * diff_val_jitter/self.jitter_radius + filt[i-1] * (1.0 - diff_val_jitter/self.jitter_radius)
            else:
                filt[i] = q[i]
            filt[i] = filt[i] * (1.0 - self.smoothing) + (filt[i-1] + trend[i-1]) * self.smoothing
            diff_jitter = filt[i] - filt[i-1]
            trend[i] = trend[i-1] * (1.0 - self.correction) + diff_jitter * self.correction
        
        return np.array(filt)