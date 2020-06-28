import matplotlib.pyplot as plt
import numpy as np

class RadarDSP:

    important_range = 1

    def __init__(self, radar_config):
        self.config = radar_config
        # Generate slow time xvals
        self.st_xvals = np.linspace(0, int(radar_config.slow_time_buffer_sec), int(radar_config.slow_time_fft_size))
        # Range FFT xvals
        self.r_xvals = np.linspace(0, int(radar_config.range_fft_size / 2), int(radar_config.range_fft_size / 2))
        self.r_xvals = self.r_xvals * radar_config.range_resolution

        self.important_bin = int(self.important_range / radar_config.range_resolution)
        self.circular_buff = np.zeros((int(radar_config.range_fft_size / 2), int(radar_config.slow_time_fft_size)), dtype=complex)

        plt.ion()

    def process_packet(self, json_data):
        real = np.array(json_data["data"]["real"])
        imag = np.array(json_data["data"]["imag"])

        complex = real + 1j * imag
        complex_abs = np.absolute(complex)

        self.circular_buff = np.roll(self.circular_buff, -1, axis=1)
        self.circular_buff[:, self.circular_buff.shape[1] - 1] = complex

        #complex_graph_mti = complex_abs - np.absolute(self.circular_buff[:, 0])
        #plt.figure(1)
        #plt.plot(self.r_xvals, complex_graph_mti)
        #plt.draw()

        plt.figure(2)
        plt.plot(self.st_xvals, self.circular_buff[self.important_bin].imag)
        plt.draw()

        plt.pause(0.00001)
        plt.figure(2)
        plt.clf()
        #plt.figure(1)
        #plt.clf()

