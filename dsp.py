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

class RadarDSP:

    important_range = 1
    bin_range = 6
    integrate_range = 2

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

        self.important_bin = int(self.important_range / radar_config.range_resolution)

        # Window Buffers
        self.circular_buff = np.zeros((int(radar_config.range_fft_size / 2), int(radar_config.slow_time_fft_size)), dtype=complex)
        self.circular_buff_mti = np.zeros((int(radar_config.range_fft_size / 2), int(radar_config.slow_time_fft_size)), dtype=complex)

        self.mti_image = np.zeros(int(radar_config.range_fft_size / 2), dtype=complex)

        self.object_distance_window_size = radar_config.frame_rate * 4
        self.object_distance_idx = 0
        self.object_distance_buff_ptp = np.zeros(int(self.object_distance_window_size))

        self.run_count = 0

        self.window_count = 0
        self.window_size = radar_config.frame_rate * 1

        self.peak_search_window_size = radar_config.frame_rate * 8

        #self.fig, self.ax = plt.subplots(self.plot_ranges * 2 + 1)
        self.fig, self.ax = plt.subplots(5)

        self.svd_denoiser = Denoiser()
        self.emd = EMD()

        self.temp_first_time_colorbar = False

        plt.ion()

        print ("FFT BR resolution: " + str(60 * self.st_resolution_hz))

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

    def graph_slow_time(self):
        exit(0)

    def process_range_bin(self, bin_num, graph_idx):
        slow_time_image = np.absolute(self.circular_buff[bin_num])

        denoised = self.svd_denoiser.denoise(slow_time_image, int(self.config.slow_time_fft_size / 16))

        # Wavelet Process
        wavelets = pywt.wavedec(denoised, 'dmey', level=4)

        # Empirical Mode Decomposition Process
        imfs = self.emd.emd(denoised)

        #self.ax[i].plot(slow_time_image)
        #i += 1
        self.ax[graph_idx].plot(denoised)

        #for n, imf in enumerate(imfs):
        #    if i >= 9:
        #        break
        #    self.ax[i].plot(imf)
        #    i += 1

        #for n, imf in enumerate(imfs):

        #for i, wavelet in enumerate(wavelets):
        #    self.ax[i + 2].plot(wavelet)

    def object_bin_estimate(self):
        # Determine object range
        min_range = self.important_bin - self.bin_range
        if (min_range < 0):
            min_range = 0

        max_range = self.important_bin + self.bin_range
        if (max_range > int(self.config.range_fft_size / 2) - 1):
            max_range = int(self.config.range_fft_size / 2) - 1

        highest_ptp_power_idx = 0
        highest_ptp_power = 0

        start_idx = int(self.config.slow_time_fft_size - self.peak_search_window_size - 1)
        end_idx = int(self.config.slow_time_fft_size - 1)
        for i in range(min_range, max_range + 1):
            power = np.absolute(np.max(self.circular_buff_mti[i][start_idx:end_idx].real) - np.min(self.circular_buff_mti[i][start_idx:end_idx].real))

            if (power > highest_ptp_power):
                highest_ptp_power = power
                highest_ptp_power_idx = i

        self.object_distance_buff_ptp[int(self.object_distance_idx)] = int(highest_ptp_power_idx)
        self.object_distance_idx += 1
        self.object_distance_idx %= self.object_distance_window_size

        estimated_range_bin_ptp = np.average(self.object_distance_buff_ptp)
        estimated_range_bin_ptp = int(round(estimated_range_bin_ptp))

        return estimated_range_bin_ptp

    def process_packet(self, json_data):
        chirp = np.array(json_data["data"]["frame"])
        complex = fft(chirp)
        complex = complex[0:int(complex.size / 2)]

        self.circular_buff = np.roll(self.circular_buff, -1, axis=1)
        self.circular_buff[:, self.circular_buff.shape[1] - 1] = complex

        # Remove clutter
        for i in range(len(self.mti_image)):
            self.mti_image[i] = complex[i] - np.average(self.circular_buff[i])

        self.circular_buff_mti = np.roll(self.circular_buff_mti, -1, axis=1)
        self.circular_buff_mti[:, self.circular_buff_mti.shape[1] - 1] = self.mti_image

        self.run_count += 1

        if (self.run_count < self.config.slow_time_fft_size):
            return

        np.savez('data', buff=self.circular_buff, buff_mti=self.circular_buff_mti)

        exit(-1)

        estimated_range_bin = self.object_bin_estimate()

        # Update graph every window count
        self.window_count += 1
        if (self.window_count < self.window_size):
            return
        else:
            self.window_count = 0

        print ("Object estimated range: " + str(estimated_range_bin * self.config.range_resolution) + "m")
        print ("Object estimated bin: " + str(estimated_range_bin))

        slow_time = self.circular_buff_mti[estimated_range_bin]

        i = 0
        self.ax[i].plot(self.st_xvals, self.circular_buff[estimated_range_bin - 2].real)
        self.ax[i].set_title("Range bin: " + str(estimated_range_bin - 2))
        i += 1

        self.ax[i].plot(self.st_xvals, self.circular_buff[estimated_range_bin - 1].real)
        self.ax[i].set_title("Range bin: " + str(estimated_range_bin - 1))
        i += 1

        self.ax[i].plot(self.st_xvals, self.circular_buff[estimated_range_bin].real)
        self.ax[i].set_title("Range bin: " + str(estimated_range_bin))

        i += 1
        self.ax[i].plot(self.st_xvals, self.circular_buff[estimated_range_bin + 1].real)
        self.ax[i].set_title("Range bin: " + str(estimated_range_bin + 1))

        i += 1
        self.ax[i].plot(self.st_xvals, self.circular_buff[estimated_range_bin + 2].real)
        self.ax[i].set_title("Range bin: " + str(estimated_range_bin + 2))

        #plt.imshow(np.absolute(self.circular_buff_mti), interpolation='nearest', aspect='auto')

        #if(self.temp_first_time_colorbar == False):
        #    plt.colorbar()
        self.temp_first_time_colorbar = True

        plt.draw()

        plt.pause(0.00001)

        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()
        self.ax[3].cla()
        self.ax[4].cla()
        #np.savez('data', self.circular_buff)

    def temp_process_packet(self, json_data):
        chirp = np.array(json_data["data"]["frame"])
        complex = fft(chirp)
        complex = complex[0:int(complex.size / 2)]

        self.circular_buff = np.roll(self.circular_buff, -1, axis=1)
        self.circular_buff[:, self.circular_buff.shape[1] - 1] = complex

        # Remove clutter
        for i in range(len(self.mti_image)):
            self.mti_image[i] = complex[i] - np.average(self.circular_buff[i])

        self.circular_buff_mti = np.roll(self.circular_buff, -1, axis=1)
        self.circular_buff_mti[:, self.circular_buff_mti.shape[1] - 1] = self.mti_image

        plt.plot(np.absolute(self.mti_image))

        plt.draw()

        plt.pause(0.00001)

        plt.cla()

        return

        #max_avg = -1
        #max_avg_index = 0

        #for i in range(self.important_bin - self.bin_range, self.important_bin + self.bin_range + 1):
        #    avg = np.average(self.circular_buff_mti[i])
        #    if (avg >= max_avg):
        #        max_avg = avg
        #        max_avg_index = i

        min_range = self.important_bin - self.plot_ranges
        if (min_range < 0):
            min_range = 0

        max_range = self.important_bin + self.plot_ranges
        if (max_range > int(self.config.range_fft_size / 2) - 1):
            max_range = int(self.config.range_fft_size / 2) - 1


        if (self.run_count == 0):
            x = 0
            for i in range(min_range, max_range + 1):
                self.ax[x].plot(self.st_xvals, np.absolute(self.circular_buff[i]))
                x += 1

            plt.draw()

            plt.pause(0.00001)

            #plt.cla()
            x = 0
            for i in range(min_range, max_range + 1):
                self.ax[x].cla()
                x += 1


        self.run_count += 1
        if (self.run_count % self.config.frame_rate == self.config.frame_rate - 1):
            self.run_count = 0

        return

        max_delta = 0
        max_delta_index = 0

        for i in range(self.important_bin - self.bin_range, self.important_bin + self.bin_range + 1):
            temp_delta = np.absolute(np.max(self.circular_buff_mti[i]) - np.min(self.circular_buff_mti[i]))
            if (temp_delta >= max_delta):
                max_delta = temp_delta
                max_delta_index = i

        slow_time_integrate = np.zeros(int(self.config.slow_time_fft_size), dtype='complex128')

        for i in range(max_delta_index - self.integrate_range, max_delta_index + self.integrate_range + 1):
            slow_time_integrate += self.circular_buff_mti[i]

        slow_time_abs = np.absolute(slow_time_integrate)

        medfilt_window = int(self.config.frame_rate / 2)
        if (medfilt_window % 2 == 0):
            medfilt_window -= 1


        #filter_slow_time = slow_time_abs

        #filter_slow_time = self.butter_lowpass_filter(slow_time_abs, 0.6, self.config.frame_rate, 1)

        slow_time_fft = np.absolute(ifft(slow_time_abs - np.average(slow_time_abs)))

        filter_slow_time = medfilt(slow_time_abs, medfilt_window)

        #filter_slow_time = savgol_filter(slow_time_abs, 11, 3)

        max_val = np.max(filter_slow_time)
        if (max_val == 0):
            max_val = 0.0000001

        indexes = peakutils.indexes(filter_slow_time, thres=0.0006/max_val, min_dist=int(self.config.frame_rate * 1.7))

        predicted_br = len(indexes) / self.config.slow_time_buffer_sec * 60

        #if (self.run_count == 0):
            #print ("Edge BR prediction: " + str(predicted_br))

        lower_limit = 0
        upper_limit = int(0.8 / self.st_resolution_hz)

        fft_max_power_idx = np.argmax(slow_time_fft[lower_limit:upper_limit])

        weighting_range = 2

        lower_limit = fft_max_power_idx - weighting_range
        if (lower_limit < 0):
            lower_limit = 0

        weighting = 0
        max_power_abs = slow_time_fft[fft_max_power_idx]

        for i in range(fft_max_power_idx - weighting_range, fft_max_power_idx + weighting_range + 1):
            power_ratio = slow_time_fft[i] / max_power_abs
            if (i < fft_max_power_idx):
                weighting -= i * power_ratio
            elif (i > fft_max_power_idx):
                weighting += i * power_ratio
            else:
                weighting += i

        breathing_hz = fft_max_power_idx * self.st_resolution_hz

        if (self.run_count == 0):
            print ("FFT BR prediction: " + str(60 * breathing_hz))

        #plt.plot(self.st_fft_xvals, slow_time_fft)
        #plt.plot(self.st_xvals, filter_slow_time)
        #plt.plot(self.st_xvals, slow_time_abs)
        plt.imshow(np.absolute(st_buffer), interpolation='nearest', aspect='auto')
        plt.colorbar()
        #plt.ylim(-0.001, 0.001)

        #if (len(indexes) > 0):
        #    plt.plot(self.st_xvals[indexes], filter_slow_time[indexes], 'xr')

        plt.draw()

        plt.pause(0.00001)

        plt.cla()

        self.run_count += 1
        if (self.run_count % self.config.frame_rate == self.config.frame_rate - 1):
            self.run_count = 0

