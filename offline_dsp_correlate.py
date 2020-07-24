import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, ifft
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import detrend
from scipy.signal import butter, lfilter
from scipy.signal import argrelextrema
from scipy.signal import cwt
from scipy.signal import ricker
from scipy.signal import correlate
import numpy as np

import peakutils
from denoise import Denoiser
from PyEMD import EMD

def butter_lowpass(cutoff, fcorrelateer=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

svd_denoiser = Denoiser()
emd = EMD()

frame_rate = 32
slow_time_buffer_sec = frame_rate * 2
slow_time_fft_size = frame_rate * slow_time_buffer_sec

st_xvals = np.linspace(0, int(slow_time_buffer_sec), int(slow_time_fft_size))

# Slow time FFT xvals
st_resolution_hz = frame_rate / slow_time_fft_size

st_fft_xvals = np.linspace(0, int(slow_time_fft_size), int(slow_time_fft_size))
st_fft_xvals *= st_resolution_hz

def get_peaks(signal):
    max_val = np.max(signal)
    if (max_val == 0):
        max_val = 0.0000001

    return peakutils.indexes(signal, thres=0.06/max_val, min_dist=int(frame_rate * 1.7))


file = open('data.txt')
lines = file.readlines()

slow_time_correlated = np.zeros(int(slow_time_fft_size))

x = 0

for line in lines:
    x += 1
    i = 0

    line = line.strip()
    str_array = np.array(line.split(', '))
    slow_time_abs = str_array.astype(np.float)

    denoised = svd_denoiser.denoise(slow_time_abs, int(slow_time_fft_size / 64))

    denoised_detrend = detrend(denoised)

    if (x == 1):
        slow_time_correlated = denoised_detrend
    else:
        slow_time_correlated = correlate(slow_time_correlated, denoised_detrend, mode="same")

    plt.figure(x)
    fig, ax = plt.subplots(3, sharex=False)

    ax[i].plot(st_xvals, slow_time_abs)
    ax[i].set_title("Raw Signals")
    i += 1
    ax[i].plot(st_xvals, slow_time_correlated)
    ax[i].set_title("Correlated Signal")
    i += 1

    max_range = int(2 / st_resolution_hz)
    fft_slow_time = np.absolute(fft(slow_time_correlated))[0:max_range]
    ax[i].plot(st_fft_xvals[0:max_range], fft_slow_time)
    ax[i].set_title("FFT Of Slow Time")
    i += 1

    print ("Iteration: " + str(x))

plt.show()


