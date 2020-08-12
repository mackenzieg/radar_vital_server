import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, ifft
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
import numpy as np
import collections

import peakutils
from denoise import Denoiser
from PyEMD import EMD
import tools

class RadarDSP:

    bin_range = 1

    def __init__(self, radar_config):
        self.update_config(radar_config)

    def update_config(self, radar_config):
        self.config = radar_config
        # Generate slow time xvals
        self.st_xvals = np.linspace(0, int(radar_config.slow_time_buffer_sec), int(radar_config.slow_time_fft_size))
        # Slow time FFT xvals
        self.st_resolution_hz = radar_config.frame_rate / radar_config.slow_time_fft_size

        # Slow time xvals
        self.st_fft_xvals = np.linspace(0, int(radar_config.slow_time_fft_size), int(radar_config.slow_time_fft_size))
        self.st_fft_xvals *= self.st_resolution_hz

        # Range FFT xvals
        self.r_xvals = np.linspace(0, int(radar_config.range_fft_size / 2), int(radar_config.range_fft_size / 2))
        self.r_xvals = self.r_xvals * radar_config.range_resolution

        # Window Buffers
        self.circular_buff = np.zeros((int(radar_config.range_fft_size / 2), int(radar_config.slow_time_fft_size)), dtype=complex)
        self.circular_buff_mti = np.zeros((int(radar_config.range_fft_size / 2), int(radar_config.slow_time_fft_size)), dtype=complex)

        self.mti_image = np.zeros(int(radar_config.range_fft_size / 2), dtype=complex)

        self.run_count = 0

        self.window_count = 0
        self.window_size = radar_config.frame_rate * 1

        self.fig, self.ax = plt.subplots((self.bin_range * 2 + 1)*2 + 1)

        self.svd_denoiser = Denoiser()
        self.emd = EMD()

        # Linear equation that gives 5 bins for 0.1m rr and 8 for 0.025m rr
        self.corr_bin_range = round(radar_config.range_resolution * (-40) + 9)

        self.prediction_idx = 0
        self.prediction_window = np.zeros(8)

        plt.ion()

        self.last_predicted_range = 0

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def graph_range_fft(self, ax):
        plt.plot(self.r_xvals, np.absolute(self.circular_buff_mti[:, self.circular_buff_mti.shape[1] - 1]))
        plt.ylim(0, 0.004)

    def process_packet(self, json_data):
        chirp = np.array(json_data["data"]["frame"])
        complex = fft(chirp)
        complex = complex[0:int(complex.size / 2)]

        self.circular_buff = np.roll(self.circular_buff, -1, axis=1)
        self.circular_buff[:, self.circular_buff.shape[1] - 1] = complex

        self.run_count += 1

        if (self.run_count < self.config.slow_time_fft_size):
            return

        self.window_count += 1
        if (self.window_count % self.window_size != self.window_size - 1):
            return

        # Remove clutter
        for i in range(len(self.mti_image)):
            self.circular_buff_mti[i] = self.circular_buff[i] - np.average(self.circular_buff[i])

        object_idx, r_value, lag, acorr, peaks = tools.auto_corr(data_matrix=self.circular_buff_mti, fs=self.config.frame_rate, delta_bin=self.corr_bin_range)

        #plt.imshow(np.abs(self.circular_buff_mti), interpolation='nearest', aspect='auto')
        #plt.draw()
        #plt.pause(0.000001)
        #plt.cla()

        x = 0
        for i in range(self.last_predicted_range - self.bin_range, self.last_predicted_range + self.bin_range + 1):
            if (i < 0 or i >= self.circular_buff_mti.shape[0]):
                continue

            self.ax[x].plot(self.st_xvals, self.circular_buff_mti[i].real)
            self.ax[x].set_title("Bin: " + str(i))
            x += 1
            filtered = tools.butter_filter(data=self.circular_buff_mti[i].real, cutoff=[1.5], fs=self.config.frame_rate, btype='low')
            self.ax[x].plot(self.st_xvals, filtered)
            self.ax[x].set_title("Bin: " + str(i) + " filtered")
            x += 1

        self.ax[x].plot(acorr)
        if (len(peaks) > 0):
            self.ax[x].plot(peaks, acorr[peaks], 'xr')
        self.ax[x].set_title("Correlation Spectrum")
        x += 1

        plt.draw()
        plt.pause(0.000001)

        x = 0
        for i in range((self.bin_range*2 + 1) * 2 + 1):
            self.ax[x].cla()
            x += 1

        self.prediction_window[self.prediction_idx] = lag
        self.last_predicted_range = object_idx

        self.prediction_window += 1
        if (self.prediction_idx % len(self.prediction_window) == len(self.prediction_window)):
            self.prediction_idx = 0

        if (np.average(self.prediction_window) == 0):
            print ("No object present")

        print (object_idx, r_value, lag)

        if (r_value < 0.5):
            return

        predicted_bpm = ((60 / (lag / self.config.frame_rate))*1.25)

        print ("Object predicted index: " + str(object_idx))
        print ("Correlation predicted bpm: " + str(predicted_bpm))


