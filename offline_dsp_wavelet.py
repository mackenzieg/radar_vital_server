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
import numpy as np

import peakutils
from denoise import Denoiser
from PyEMD import EMD

def butter_lowpass(cutoff, fs, order=5):
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

slow_time_buffer_sec = 32 * 2
frame_rate = 32
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

slow_time_abs_integ = np.zeros(int(slow_time_fft_size))

x = 0
for line in lines:
    x += 1

    line = line.strip()
    str_array = np.array(line.split(', '))
    slow_time_abs = str_array.astype(np.float)

    slow_time_abs_integ += slow_time_abs

    denoised = svd_denoiser.denoise(slow_time_abs, int(slow_time_fft_size / 16))

    denoised_detrend = detrend(denoised)

    # Wavelet Process
    wavelets = pywt.wavedec(denoised_detrend, 'dmey', level=2)

    filter_slow_time = butter_lowpass_filter(denoised_detrend, 1, frame_rate, 5)

    # Empirical Mode Decomposition Process
    imfs = emd.emd(denoised_detrend)

    plt.figure(x)

    widths = np.arange(1, 256)

    cwt_coefs = cwt(slow_time_abs, ricker, widths)

    plt.imshow(cwt_coefs, extent=[-1, 1, 1, 31], cmap="PRGn", aspect='auto',
            vmax=abs(cwt_coefs).max(), vmin=-abs(cwt_coefs).max())



    print ("Iteration: " + str(x))

plt.show()


