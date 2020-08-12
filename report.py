import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, ifft
from scipy.signal import medfilt
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import detrend
from scipy.signal import butter, lfilter
from scipy.signal import argrelextrema
from scipy.signal import correlate
from scipy.signal import cwt
from scipy.stats import linregress
import numpy as np

import peakutils
from denoise import Denoiser
from PyEMD import EMD

from collections import deque


fig, ax = plt.subplots(2)


duration = 1024

t = np.arange(duration)
f0 = 0.5
fs = 32

#signal = np.sin(2*np.pi*t*(f0 / fs))
#signal += np.random.normal(0, 1, duration)
signal = np.random.normal(0, 0.1, duration)

signal -= np.mean(signal)

n = signal.size

result = np.correlate(signal, signal, mode='same')

# Center autocorrelation coefficients with lag 1 to index 0
acorr = result[n//2 + 1:] / (signal.var() * np.arange(n-1, n//2, -1))

# Mirror array to perform more accurate peak detection
mirror_acorr = np.concatenate((acorr[::-1], acorr))

length = len(acorr)
x_test = np.arange(0, length)

# Possible breathing rates to test
min_br_hz = fs // 0.05 # 3 bpm
max_br_hz = fs // 0.8 # 48 bpm
#ranges = np.arange(frame_rate // max_br_hz - 1, frame_rate // min_br_hz)

# Find peaks of highest correlation coefficients
#peaks = find_peaks_cwt(mirror_acorr, ranges)
peaks = find_peaks(mirror_acorr, distance=max_br_hz-1, prominence=0.4)

# Remove peaks with lag time below fs
#peaks = peaks[peaks > frame_rate]

#print ("Index: " + str(x))
#print (peaks)

slope, intercept, r_value, p_value, std_err = linregress(x_test, acorr)

print ("r_value: " + str(r_value))

mirror = True

ax[0].plot(t, signal)
if (mirror):
    pks = peaks[0]
    #ax[1].plot(t[:len(mirror_acorr)], mirror_acorr)
    #ax[1].plot(pks, mirror_acorr[pks], 'xr')
    ax[1].plot(acorr)
    ax[1].set_ylim([-0.2, 0.8])

else:
    ax[1].plot(t, result)

plt.show()
