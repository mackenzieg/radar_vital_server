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
    for i in range(1, st_buffer_mti.shape[0]):
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

import glob

min_bin = 31
max_bin = 0

prec_bins = []

for file in glob.glob("doll_data/data*.npz"):
    #file_path = "server_data/data1.npz"
    print ("Processing file: " +str(file))

    st_buffer = np.load(file)['buff']
    st_buffer_mti = np.load(file)['buff_mti']

    for i in range(st_buffer.shape[0]):
        st_buffer_mti[i] = st_buffer[i] - np.average(st_buffer[i])

    corr_idx = tools.auto_corr(st_buffer_mti, 32)
    print ("Correlation prediction: " + str(corr_idx))

    bin = corr_idx[0]

    if (min_bin > bin):
        min_bin = bin

    if (max_bin < bin):
        max_bin = bin

    prec_bins.append(corr_idx[0])


    fig, ax = plt.subplots(7)

    range_bin = 3
    i = 0
    for x in range(corr_idx[0] - range_bin, corr_idx[0] + range_bin + 1):
        ax[i].plot(st_xvals, st_buffer_mti[x].real)
        ax[i].set_title("Bin: " + str(x))
        i += 1

    plt.show()

    #plt.xlabel("Time (s)")

    #plt.show()

#plt.hist(prec_bins, bins=(max_bin - min_bin))
print (prec_bins)

