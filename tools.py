from scipy.signal import butter, lfilter
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

def auto_corr(st_buffer_mti, delta_bin=5):
    max_corr = 0
    max_corr_idx = 0

    object_bin = track_target(st_buffer_mti)
    print (object_bin)

    for i in range(object_bin - delta_bin, object_bin + delta_bin + 1):
        signal = st_buffer_mti[i].real
        signal -= np.mean(signal)
        n = signal.size

        result = np.correlate(signal, signal, mode='same')

        acorr = result[n//2 + 1:] / (signal.var() * np.arange(n-1, n//2, -1))

        lag = np.abs(acorr).argmax() + 1

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
        slow_time_abs = np.absolute(st_buffer_mti[:, x])
        max_idx = slow_time_abs.argmax()

        if (max_idx == object_idx):

            if (confidence < max_confidence):
                confidence += 1

            continue

        confidence -= 1
        if (confidence <= 0):
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

