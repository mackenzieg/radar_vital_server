from scipy.signal import butter, lfilter
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
from scipy.stats import linregress
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

def peak_detect(data, fs, method='find_peaks', min_bpm=6, max_bpm=48, prominence=0.4):
    # Possible breathing rates to test
    max_br_samples = fs * (60 // max_bpm) # Convert bpm to samples per breath

    peaks = []
    if (method == 'find_peaks'):
        peaks = find_peaks(data, distance=max_br_samples-1, prominence=prominence)[0]
    else:
        max_br_samples = fs * (60 // min_bpm) + 1
        ranges = np.arange(min_br_samples, max_br_samples)

        peaks = find_peaks_cwt(data, ranges)

    return peaks


def auto_corr(st_buffer_mti, fs, delta_bin=6, lin_corr_thres=0.6):
    svd_denoiser = Denoiser()
    max_corr = 0
    max_corr_idx = 0

    object_bin = track_target(st_buffer_mti)

    print ("Starting with prediction: " + str(object_bin))

    min_range = object_bin - delta_bin
    if (min_range <= 0):
        min_range = 1

    max_range = object_bin + delta_bin
    if (max_range >= st_buffer_mti.shape[0]):
        max_range = st_buffer_mti.shape[0]

    x_vals = np.arange(1, st_buffer_mti.shape[1] // 2)

    for i in range(min_range, max_range):
        signal = st_buffer_mti[i].real
        signal -= np.mean(signal)
        n = signal.size

        result = np.correlate(signal, signal, mode='same')

        # Center autocorrelation coefficients with lag 1 to index 0
        acorr = result[n//2 + 1:] / (signal.var() * np.arange(n-1, n//2, -1))

        slope, intercept, r_value, p_value, std_err = linregress(x_vals, acorr)

        print ("Index: " + str(i))
        print ("Lin coeef:" + str(r_value**2))

        # Make sure result isn't a linear response
        if (r_value**2 > lin_corr_thres):
            continue

        # Mirror array to perform more accurate peak detection
        mirror_acorr = np.concatenate((acorr[::-1], acorr))

        # Find peaks of highest correlation coefficients
        #peaks = find_peaks_cwt(acorr, np.arange(1, fs * 8))
        peaks = peak_detect(mirror_acorr, fs=fs)

        if (peaks.size == 0):
            continue

        print (peaks)

        idx_max_peak = 0
        if (len(peaks) <= 1):
            # Not enough peaks detected to perform analysis
            continue

        if (len(peaks) % 2 == 0):
            # Peak count is even meaning peak detection didn't detect harmonics
            continue

        middle_idx = len(peaks) // 2
        # Always take first harmonic (lag > 1)
        idx_max_peak = peaks[middle_idx + 1]

        # Recenter peak index relative to acorr due to mirroring process
        idx_max_peak -= len(acorr)

        # First peak is lag time
        lag = idx_max_peak + 1

        r = acorr[lag - 1]

        if (r > max_corr):
            max_corr = r
            max_corr_idx = i

    return max_corr_idx, max_corr, lag

def track_target(st_buffer_mti):
    confidence = 0
    object_idx = 0

    range_fft_size = st_buffer_mti.shape[0]
    slow_time_fft_size = st_buffer_mti.shape[1]

    max_confidence = 32

    for x in range(slow_time_fft_size):
        slow_time_abs = np.absolute(st_buffer_mti[1:, x])
        max_idx = slow_time_abs.argmax()

        if (max_idx == object_idx):

            if (confidence < max_confidence):
                confidence += 1

            continue

        confidence -= 1
        if (confidence <= 0):
            confidence = 1
            object_idx = max_idx

    return object_idx


def resp(signal, sampling_rate=32):
    filtered = butter_filter(data=signal, cutoff=[0.1, 0.5], fs=sampling_rate, btype='band')

    zeros = zero_cross(signal=filtered, detrend=True)
    beats = zeros[::2]

    if len(beats) < 2:
        print ("This guy dead lol")
        return None

    rate_idx = beats[1:]
    rate = sampling_rate * (1. / np.diff(beats))

    indx = np.nonzero(rate <= 0.5)
    rate_idx = rate_idx[indx]
    rate = rate[indx]

    return filtered, zeros, rate

