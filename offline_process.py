import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, ifft
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import detrend
from scipy.signal import butter, lfilter
from scipy.signal import argrelextrema
from scipy.signal import correlate
from scipy.signal import cwt
import numpy as np

import peakutils
from denoise import Denoiser
from PyEMD import EMD

from collections import deque

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

import tools


svd_denoiser = Denoiser()
emd = EMD()

slow_time_buffer_sec = 32 * 1
frame_rate = 32
slow_time_fft_size = frame_rate * slow_time_buffer_sec

st_xvals = np.linspace(0, int(slow_time_buffer_sec), int(slow_time_fft_size))

# Slow time FFT xvals
st_resolution_hz = frame_rate / slow_time_fft_size

st_fft_xvals = np.linspace(0, int(slow_time_fft_size), int(slow_time_fft_size))
st_fft_xvals *= st_resolution_hz

def std_windowed(st_buffer_mti):
    window_size = 8
    upper_threshold = 0.005

    max_sd = 0
    max_sd_idx = 0
    for i in range(st_buffer_mti.shape[0]):
        signal = st_buffer[i]

        stds = np.zeros(int(slow_time_fft_size / window_size))

        signal_split = np.split(signal, slow_time_fft_size / window_size)

        x = 0
        for split in signal_split:
            stds[x] = np.std(split, ddof=1)
            x += 1

        if ((stds > upper_threshold).all()):
            continue

        sd = np.std(signal)
        if (max_sd < sd):
            max_sd = sd
            max_sd_idx = i

        #filtered, zeros, rate = tools.resp(signal, frame_rate)
        #print (list(zeros))
        #print (list(rate))

        #if ((stds < upper_threshold).all() and np.std(signal) > std_lower_theshold):
        #    return i


        #power = np.absolute(np.max(signal) - np.min(signal))

        #if (power > max_ptp_power):
        #    max_ptp_power = power
        #    max_ptp_idx = i

    return max_sd_idx

def get_peaks(signal):
    max_val = np.max(signal)
    if (max_val == 0):
        max_val = 0.0000001

    return peakutils.indexes(signal, thres=0.06/max_val, min_dist=int(frame_rate * 1.7))


file = open('data.txt')
lines = file.readlines()

st_buffer = np.load('data.npz')['buff']
st_buffer_mti = np.load('data.npz')['buff_mti']

for i in range(st_buffer.shape[0]):
    st_buffer_mti[i] = st_buffer[i] - np.average(st_buffer[i])

#max_ptp_idx = 0
#max_ptp_power = -10
#for i in range(st_buffer.shape[0]):
#    signal = st_buffer_mti[i]
#    print ("Range " + str(i) + " std: " + str(np.std(signal)))
#
#    power = np.absolute(np.max(signal) - np.min(signal))
#
#    if (power > max_ptp_power):
#        max_ptp_power = power
#        max_ptp_idx = i

max_ptp_idx = std_windowed(st_buffer_mti)
print ("Windowd std: " + str(max_ptp_idx))

corr_idx = tools.auto_corr(st_buffer_mti)
print ("Correlation: " + str(corr_idx))

max_ptp_idx = tools.track_target(st_buffer_mti)
print ("Predicted range: " + str(max_ptp_idx))

slow_time = st_buffer[max_ptp_idx + 2]


plt.imshow(np.absolute(st_buffer_mti), interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xlabel("Time (s)")
plt.ylabel("Bin number")

fig, ax = plt.subplots(5)

range_bin = 3
i = 0
for x in range(max_ptp_idx, max_ptp_idx + range_bin + 2):
    ax[i].plot(st_xvals, st_buffer_mti[x].real)
    ax[i].set_title("Real bin: " + str(x))
    i += 1

#for x in range(max_ptp_idx + 1, max_ptp_idx + range_bin + 2):
#    result = np.correlate(st_buffer_mti[i].real, st_buffer_mti[i].real, mode='full')
#    result = result[int(result.size/2):]
#    ax[i].plot(result[5:])
#    ax[i].set_title("Real bin: " + str(x) + " autocorellated")
#    i += 1

#for x in range(max_ptp_idx, max_ptp_idx + range_bin + 2):
#    siga = st_buffer_mti[x].real
#    sigb = st_buffer_mti[x + 1].real
#
#    print ("Correlation of " + str(x) + " and " + str(x + 1))
#    print (np.corrcoef(siga, sigb)[0][1])

#for x in range(max_ptp_idx - range_bin, max_ptp_idx + range_bin + 1):
#    ax[i].plot(np.diff(st_buffer_mti[x].real))
#    ax[i].set_title("Real bin: " + str(x) + " diff")
#    i += 1

filtered, zeros, rate = tools.resp(slow_time, frame_rate)
print ("Resp rates: ")
print (list(rate))

num_steps = 512
scales = np.arange(1, num_steps + 1)
wavelet_type = 'morl'
coefs, freqs = pywt.cwt(slow_time.real, scales, wavelet_type, 1/frame_rate)
plt.matshow(coefs)

#ax[i].plot(st_xvals, slow_time_denoised.real)
#ax[i].set_title("Real Denoised")
#i += 1

#ax[i].plot(st_xvals, slow_time.imag)
#ax[i].set_title("Imaginary")
#i += 1
#
#ax[i].plot(st_xvals, imag_denoised)
#ax[i].set_title("Imaginary Denoised")
#i += 1

#max_range = int(2 / st_resolution_hz)
#fft_slow_time = np.absolute(fft(slow_time_denoised))[0:max_range]
#ax[i].plot(st_fft_xvals[0:max_range], fft_slow_time)
#ax[i].set_title("FFT Of Slow Time")
#i += 1

plt.xlabel("Time (s)")
plt.show()

