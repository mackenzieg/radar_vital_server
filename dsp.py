import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import numpy as np

class RadarDSP:

    important_range = 1
    bin_range = 5

    def __init__(self, radar_config):
        self.config = radar_config
        # Generate slow time xvals
        self.st_xvals = np.linspace(0, int(radar_config.slow_time_buffer_sec), int(radar_config.slow_time_fft_size))
        # Range FFT xvals
        self.r_xvals = np.linspace(0, int(radar_config.range_fft_size / 2), int(radar_config.range_fft_size / 2))
        self.r_xvals = self.r_xvals * radar_config.range_resolution

        self.important_bin = int(self.important_range / radar_config.range_resolution)
        self.circular_buff = np.zeros((int(radar_config.range_fft_size / 2), int(radar_config.slow_time_fft_size)), dtype=complex)

        self.last_image_mti = np.zeros(int(radar_config.range_fft_size / 2), dtype=complex)

        plt.ion()

    def process_packet(self, json_data):
        real = np.array(json_data["data"]["real"])
        imag = np.array(json_data["data"]["imag"])

        complex = real + 1j * imag

        self.circular_buff = np.roll(self.circular_buff, -1, axis=1)
        self.circular_buff[:, self.circular_buff.shape[1] - 1] = complex

        #complex_graph_mti = np.absolute(self.circular_buff[:, self.circular_buff.shape[1] - 1] - self.circular_buff[:, self.circular_buff.shape[1] - 4])
        #complex_graph_mti = np.absolute(self.circular_buff[:, self.circular_buff.shape[1] - 1])
        #plt.figure(1)
        #plt.plot(self.r_xvals, complex_graph_mti)

        #plt.ylim(0, 0.004)
        #plt.draw()

        #for i in range(self.important_bin - self.bin_range, self.important_bin + self.bin_range):
        #    slow_time_fft = ifft(self.circular_buff[i])
        #    plt.plot(self.st_xvals, np.absolute(slow_time_fft))


        #plt.figure(2)

        max_avg = 0
        max_avg_index = 0
        for i in range(self.important_bin - self.bin_range, self.important_bin + self.bin_range):
            avg = np.average(self.circular_buff[i])
            if (avg >= max_avg):
                max_avg = avg
                max_avg_index = i

        #plt.plot(self.st_xvals, np.absolute(self.circular_buff[max_avg_index]))
        slow_time_abs = np.absolute(self.circular_buff[max_avg_index])

        savgol_filter_slow_time = savgol_filter(slow_time_abs, 9, 3)

        #filter_slow_time = medfilt(slow_time_abs)

        #plt.plot(self.st_xvals, filter_slow_time)
        plt.plot(self.st_xvals, savgol_filter_slow_time)
        #plt.plot(self.st_xvals, fft(savgol_filter_slow_time))
        plt.draw()

        plt.pause(0.00001)
        #plt.figure(2)
        #plt.clf()
        #plt.figure(1)
        plt.clf()

