import numpy as np
from scipy.interpolate import UnivariateSpline
from numpy.fft import fft, rfft, fftfreq
import pylab

def crude_lift_frequency(data, plot=False):
    time = np.array(data['Time'])
    samples = 100000
    retime = np.linspace(time[0], time[-1], samples)
    raw = np.array(data['HandLeft'])[:,2] + np.array(data['HandRight'])[:,2]
    smoothed = UnivariateSpline(time, raw, s=0)
    resampled = smoothed(retime)
    h = np.hanning(samples)
    freq = fftfreq(resampled.size, d=(retime[-1]-retime[0])/(60.0*samples))
    ft = fft(resampled - resampled.mean())
    if plot:
        pylab.figure("Fourier transform of hands z sum")
        pylab.plot(freq, np.abs(ft))
        pylab.axis([0,30,0,20000])
        pylab.figure("Resampled love")
        pylab.plot(retime, resampled)
        pylab.show()
    return ft
    
if __name__ == '__main__':
    import json
    with open("Web/json/6l.json","rb") as file:
        j = json.load(file)
        crude_lift_frequency(j['kin'], plot=True)