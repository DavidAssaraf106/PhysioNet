from scipy.signal import iirnotch, lfilter
from scipy.signal import butter


def notch_filter(data, cutoff_frequency, signal_quality, fs):
    b, a = iirnotch(cutoff_frequency, signal_quality, fs)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def prefiltering(ecg, signal_freq):
    ecg_bandpassed = bandpass_filter(ecg, 0.1, 90, signal_freq, 3)
    ecg_bandpassed_2 = bandpass_filter(ecg_bandpassed, 0.1, 90, signal_freq, 3)
    ecg_bandpassed_3 = bandpass_filter(ecg_bandpassed_2, 0.1, 90, signal_freq, 3)
    ecg_notch_1 = notch_filter(ecg_bandpassed_3, 50, 17, signal_freq)
    ecg_notch_2 = notch_filter(ecg_notch_1, 60, 17, signal_freq)
    return ecg_notch_2




