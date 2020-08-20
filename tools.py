from scipy.signal import butter, lfilter
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.fftpack import fft
from scipy.signal import detrend
import numpy as np

def _butter_filter(cutoff, btype, fs, order=5):
    nyq = 0.5 * fs
    cutoff = np.array(cutoff)
    cutoff /= nyq
    return butter(order, cutoff, btype=btype)

def butter_filter(data, cutoff, fs, btype='band', order=2):
    b, a = _butter_filter(cutoff, btype, fs, order)

    return lfilter(b, a, data)

def zero_cross(signal, detrend=False):
    if (detrend):
        signal = signal - np.mean(signal)

    df = np.diff(np.sign(signal))
    zeros = np.nonzero(np.abs(df) > 0)[0]

    return zeros

from denoise import Denoiser

def peak_detect(data, fs, min_rpm=6, max_rpm=35, prominence=0.4, method='find_peaks'):
    # Possible breathing rates to test
    max_br_samples = fs * (60 // max_rpm) # Convert bpm to samples per breath

    peaks = []
    if (method == 'find_peaks'):
        peaks = find_peaks(data, distance=max_br_samples-1, prominence=prominence)[0]
    elif (method = 'find_peaks_cwt'):
        max_br_samples = fs * (60 // min_rpm) + 1
        ranges = np.arange(min_br_samples, max_br_samples)

        peaks = find_peaks_cwt(data, ranges)

    return peaks

def fft_bpm(data, fs, filter_cutoff=None, min_rpm=6, max_rpm=35, use_imag=True):
    N = len(data)

    f_resolution = fs / N

    lower_idx = int((min_rpm / 60) // f_resolution)
    upper_idx = int((max_rpm / 60) // f_resolution + 1)

    if (not use_imag):
        data = data.real

    if (filter_cutoff is not None):
        data = butter_filter(data=data, cutoff=filter_cutoff, fs=fs, btype='low')

    data_ac = data - np.average(data)

    yf = fft(data_ac)

    # Remove unwanted ranges
    yf_range = np.abs(yf[lower_idx:upper_idx])

    max_idx = np.argmax(yf_range)

    predicted_bpm = max_idx * f_resolution * 60

    return predicted_bpm, yf

def auto_corr(data_matrix, fs, starting_prediction=None, delta_bin=7, lin_corr_thres=0.5, filter_corr_vals=True, corr_filter_cutoff=3.0, min_rpm=6, max_rpm=35, min_lin_r=0.5, pre_corr_processing=True):
    max_corr_idx = 0
    max_corr = 0
    max_lag = 0

    if (starting_prediction is None):
        object_bin = track_target(data_matrix)
    else:
        object_bin = starting_prediction

    min_range = object_bin - delta_bin
    if (min_range <= 0):
        min_range = 1

    max_range = object_bin + delta_bin
    if (max_range >= data_matrix.shape[0]):
        max_range = data_matrix.shape[0]

    x_vals = np.arange(1, data_matrix.shape[1] // 2)

    max_acorr = []
    max_peaks = []

    for i in range(min_range, max_range):
        signal = data_matrix[i].real

        if (pre_corr_processing):
            signal = butter_filter(data=signal, cutoff=[4.0], fs=fs, btype='low')

        # Remove DC bias
        signal = detrend(signal)
        n = signal.size

        signal_variance = signal.var()

        # Remove signals with no variance
        if (signal_variance == 0):
            continue

        result = np.correlate(signal, signal, mode='same')

        # Center autocorrelation coefficients with lag 1 to index 0
        acorr = result[n//2 + 1:] / (signal_variance * np.arange(n-1, n//2, -1))

        # Perform filtering on coefficient spectrum if option enabled
        if (filter_corr_vals):
            acorr = butter_filter(acorr, cutoff=[corr_filter_cutoff], fs=fs, btype='low')

        slope, intercept, r_value, p_value, std_err = linregress(x_vals, acorr)

        # Make sure result isn't a linear response
        if (np.abs(r_value) >= min_lin_r):
            continue

        # Mirror array to perform more accurate peak detection
        mirror_acorr = np.concatenate((acorr[::-1], acorr))

        # Find peaks of highest correlation coefficients
        peaks = peak_detect(mirror_acorr, fs=fs, min_rpm=min_rpm, max_rpm=max_rpm)

        # Not enough peaks detected to perform analysis
        if (len(peaks) <= 1):
            continue

        # Peak count is even meaning peak detection didn't detect harmonics
        if (len(peaks) % 2 == 0):
            continue

        # Idx should correspond to the first harmonic
        middle_idx = len(peaks) // 2

        # Always take first harmonic (lag > 1)
        idx_max_peak = peaks[middle_idx + 1]

        r = mirror_acorr[idx_max_peak]

        # First peak is lag time
        lag = idx_max_peak + 1
        # Recenter peak index relative to acorr due to mirroring process
        lag -= len(mirror_acorr) // 2

        # Min BR of 5 rpm
        max_allowable_lag = fs / (min_rpm / 60)
        min_allowable_lag = fs / (max_rpm / 60)
        if (lag > max_allowable_lag or lag < min_allowable_lag):
            continue

        if (r > max_corr):
            max_corr = r
            max_corr_idx = i
            max_lag = lag
            max_acorr = mirror_acorr
            max_peaks = peaks

    return max_corr_idx, max_corr, max_lag, max_acorr, max_peaks

def track_target(data_matrix, dc_removed=False):
    confidence = 0
    object_idx = 0

    range_fft_size = data_matrix.shape[0]
    slow_time_fft_size = data_matrix.shape[1]

    if (not dc_removed):
        for i in range(range_fft_size):
            data_matrix[i] = data_matrix[i] - np.average(data_matrix[i])

    max_confidence = 32

    for x in range(slow_time_fft_size):
        slow_time_abs = np.absolute(data_matrix[1:, x])
        # Add once since index 0 is removed
        max_idx = slow_time_abs.argmax() + 1

        if (max_idx == object_idx):

            if (confidence < max_confidence):
                confidence += 1

            continue

        confidence -= 1
        if (confidence <= 0):
            confidence = 1
            object_idx = max_idx

    return object_idx

def proximity_correlation(data_matrix, idx, range=1):
    min_range = idx - range
    if (min_range <= 0):
        min_range = 0

    max_range = idx + range
    if (max_range >= data_matrix.shape[0]):
        max_range = data_matrix.shape[0]

    r = []
    for i in range(min_range, max_range):
        slow_time_a = data_matrix[i + 1]
        slow_time_b = data_matrix[i]
        r.append(np.corrcoef(slow_time_a, slow_time_b)[0, 1])

    return r

def resp(signal, sampling_rate=32):
    filtered = butter_filter(data=signal, cutoff=[0.1, 0.5], fs=sampling_rate, btype='band')

    zeros = zero_cross(signal=filtered, detrend=True)
    beats = zeros[::2]

    if len(beats) < 2:
        return None

    rate_idx = beats[1:]
    rate = sampling_rate * (1. / np.diff(beats))

    indx = np.nonzero(rate <= 0.5)
    rate_idx = rate_idx[indx]
    rate = rate[indx]

    return filtered, zeros, rate, rate_idx

