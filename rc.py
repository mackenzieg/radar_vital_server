
class RadarConfig:

    # Time in seconds to buffer slow time data
    slow_time_buffer_sec = 8

    def __init__(self, radar_config_json):
        self.update_vals(radar_config_json)

    def update_vals(self, config):

        self.range_resolution           = config["range_resolution"]
        self.maximum_range              = config["maximum_range"]
        self.minimum_range              = config["minimum_range"]
        self.speed_resolution           = config["speed_resolution"]
        self.maximum_speed              = config["maximum_speed"]

        self.frame_rate                 = config["frame_rate"]
        self.adc_samplerate_hz          = config["adc_samplerate_hz"]
        self.bgt_tx_power               = config["bgt_tx_power"]
        self.rx_antenna_number          = config["rx_antenna_number"]
        self.if_gain_db                 = config["if_gain_db"]

        self.fmcw_center_frequency_khz  = config["fmcw_center_frequency_khz"]
        self.lower_frequency_khz        = config["lower_frequency_khz"]
        self.upper_frequency_khz        = config["upper_frequency_khz"]

        self.range_fft_size             = config["range_fft_size"]
        self.num_samples_per_chirp      = config["num_samples_per_chirp"]


        self.bandwidth_khz = self.upper_frequency_khz - self.lower_frequency_khz
        self.range_value_per_bin = 300000 / (self.bandwidth_khz * 2 * (self.range_fft_size / self.num_samples_per_chirp))

        self.slow_time_fft_size = self.slow_time_buffer_sec * self.frame_rate

