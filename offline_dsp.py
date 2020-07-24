import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, ifft
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import detrend
from scipy.signal import butter, lfilter
from scipy.signal import argrelextrema
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

    fig, ax = plt.subplots(8, sharex=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Power")

    i = 0
    #ax[i].plot(st_xvals, detrend(slow_time_abs))
    #ax[i].set_title("Slow Time Raw")
    #i += 1

    #ax[i].plot(st_xvals, denoised_detrend)
    #ax[i].set_title("Singular Value Decomposition")
    #peaks = argrelextrema(denoised_detrend, comparator=np.greater, order=frame_rate)
    #if (len(peaks) > 0):
    #    ax[i].plot(st_xvals[peaks], denoised_detrend[peaks], 'xr')
    #i += 1

    ax[i].plot(st_xvals, denoised_detrend)
    ax[i].set_title("Singular Value Decomposition")
    peaks = get_peaks(denoised_detrend)
    if (len(peaks) > 0):
        ax[i].plot(st_xvals[peaks], denoised_detrend[peaks], 'xr')
    i += 1

    #ax[i].plot(st_xvals, denoised_detrend)
    #ax[i].set_title("Singular Value Decomposition With Detrend (Linear)")
    #peaks = get_peaks(denoised_detrend)
    #if (len(peaks) > 0):
    #    ax[i].plot(st_xvals[peaks], denoised_detrend[peaks], 'xr')
    #i += 1

    #ax[i].plot(st_xvals, filter_slow_time)
    #ax[i].set_title("Bandpass Filter")
    #i += 1

    #max_range = int(2 / st_resolution_hz)
    #fft_slow_time = np.absolute(fft(detrend(slow_time_abs)))[0:max_range]
    #ax[i].plot(st_fft_xvals[0:max_range], fft_slow_time)
    #ax[i].set_title("FFT Of Slow Time")
    #i += 1

    #ax[0].plot(st_xvals[:len(filter_slow_time)//2], filter_slow_time[:len(filter_slow_time)//2])
    #ax[1].plot(st_xvals[len(filter_slow_time)//2:], filter_slow_time[len(filter_slow_time)//2:])

    for imf in imfs:
        if i > 5:
            break
        ax[i].plot(st_xvals, imf)
        ax[i].set_title("EMD")
        i += 1

        max_range = int(2 / st_resolution_hz)
        fft_slow_time = np.absolute(fft(detrend(imf)))[0:max_range]
        ax[i].plot(st_fft_xvals[0:max_range], fft_slow_time)
        ax[i].set_title("FFT Of EMD")
        i += 1

    #packet_lvl = 1
    #for wavelet in wavelets:
    #    ax[i].plot(wavelet)
    #    ax[i].set_title("Wavelet Decomposition Level " + str(packet_lvl))
    #    i += 1
    #    packet_lvl += 1


    print ("Iteration: " + str(x))

    #max_idx = np.argmax(fft_slow_time)
    #print ("Predicted rpm: " + str(max_idx * st_resolution_hz * 60))

#fig, ax = plt.subplots(4, sharex=False)
#
#denoised = svd_denoiser.denoise(slow_time_abs_integ, int(slow_time_fft_size / 16))
#
#denoised_detrend = detrend(denoised)
#
#
#filter_slow_time = butter_lowpass_filter(denoised_detrend, 1, frame_rate, 5)
#
#i = 0
#ax[i].plot(st_xvals, slow_time_abs_integ)
#ax[i].set_title("Slow Time Raw")
#i += 1
#ax[i].plot(st_xvals, denoised_detrend)
#ax[i].set_title("Singular Value Decomposition")
#i += 1
#ax[i].plot(st_xvals, filter_slow_time)
#ax[i].set_title("Lowpass Filter")
#i += 1
#
#max_range = int(2 / st_resolution_hz)
#fft_slow_time = np.absolute(fft(filter_slow_time))[0:max_range]
#ax[i].plot(st_fft_xvals[0:max_range], fft_slow_time)
#ax[i].set_title("FFT Of Slow Time")
#i += 1

#
#wavelets = pywt.wavedec(filter_slow_time, 'dmey', level=3)
#packet_lvl = 1
#for wavelet in wavelets:
#    ax[i].plot(wavelet)
#    ax[i].set_title("Wavelet Decomposition Level " + str(packet_lvl))
#    i += 1
#    packet_lvl += 1

plt.show()


