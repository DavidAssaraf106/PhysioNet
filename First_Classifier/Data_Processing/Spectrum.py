import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import get_window
from scipy import fftpack
import csv
from scipy.signal import windows
from skimage import util
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import savgol_filter, butter, lfilter


warnings.filterwarnings("ignore")
input_directory = "/home/david/Training_WFDB_new"


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def get_freq(header_data):
    for lines in header_data:
        tmp = lines.split(' ')
        return tmp[2]


# The Welch method: estimates directly the Power spectrum: in V**2 (in order for us to compute the log-domain quantity)
# via an average over periodograms
def get_PSD_decomposition(ecg_lead, freq):
    n = len(ecg_lead)
    power_2 = int(np.log(n) / np.log(2) + 1)
    freqs, psd = signal.welch(ecg_lead, fs=freq, window='hamming', nperseg=2 ** power_2, scaling='spectrum')
    return freqs, psd


def harmonic_spectrum_decomposition(ecg_lead, freq):
    n = len(ecg_lead)
    frequencies = np.asarray(
        fftpack.fft(ecg_lead)) ** 2  # todo: first convolve the signal with a window function/ WOULD BE BETTER
    # you will use power_2 = np.log(n)/np.log(2)+1
    freqs = fftpack.fftfreq(len(ecg_lead)) * freq
    power = np.abs(frequencies) / n
    return freqs, power


def frequency_over_time(ecg_lead, freq):
    n = len(ecg_lead)
    power_2 = np.log(n) / np.log(2) + 1
    slices = util.view_as_windows(ecg_lead, window_shape=(2 ** int(power_2 - 1),), step=100)
    M = 2 ** int(power_2 - 1) + 1
    win = np.hamming(M)[:-1]
    slices = slices * win
    slices = slices.T
    spectrum = np.fft.fft(slices, axis=0)[: M // 2 + 1:-1]
    spectrum = np.abs(spectrum)
    S = np.abs(spectrum)
    S = 10 * np.log10(S / np.max(S))
    return S


def frequency_over_time_normalized(ecg_lead, freq):  # todo: change le overlap?
    n = len(ecg_lead)
    power_2 = np.log(n) / np.log(2) + 1
    freqs, times, Sx = signal.spectrogram(ecg_lead, fs=freq, window='hamming',
                                          nperseg=1024, noverlap=100,
                                          detrend=False, scaling='spectrum')
    Sx = 10 * np.log10(Sx / np.max(Sx))
    return Sx, times


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def filtering_method(ecg_lead):
    ecg_filtered = bandpass_filter(ecg_lead, 0.3, 75, 500, 3)
    ecg_filtered = bandpass_filter(ecg_filtered, 0.3, 75, 500, 3)
    ecg_smoothed = savgol_filter(ecg_filtered, window_length=17, polyorder=4, deriv=0)
    ecg_smoothed = savgol_filter(ecg_smoothed, window_length=17, polyorder=4, deriv=0)
    return ecg_smoothed





if __name__ == '__main__':
    experiment_1 = ''
    input_files = []
    i = 0
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
            i += 1
    input_files.sort()
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for i, f in enumerate(input_files):
    tmp_input_file = os.path.join(input_directory, f)
    data, header_data = load_challenge_data(tmp_input_file)
    patient = header_data[0].split(' ')[0]
    freq = int(get_freq(header_data))
    for i_lead, lead in enumerate(list_lead):
        f, axarr = plt.subplots(1, 2, figsize=(25, 10))
        ecg_lead = data[lead]
        ecg_filtered = bandpass_filter(ecg_lead, 0.05, 100, 500, 3)
        ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 48, 52, 500, 3)
        ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 58, 62, 500, 3)
        freqs_harmonic, power_harmonic = harmonic_spectrum_decomposition(ecg_lead, freq)
        axarr[0] = plt.subplot(211)
        axarr[0].plot(np.arange(len(ecg_lead))/freq, ecg_lead, label='Raw ecg',
                      color='g')
        axarr[0].set_xlabel('Time [s]')
        axarr[0].set_ylabel('Amplitude [mV]')
        axarr[0].set_title('Raw ECG')
        axarr[0].legend()
        axarr[1] = plt.subplot(212)
        axarr[1].plot(np.arange(len(ecg_lead))/freq, ecg_filtered, label='Filtered ecg',
                      color='g')
        axarr[1].set_xlabel('Time [s]')
        axarr[1].set_ylabel('Amplitude [mV]')
        axarr[1].set_title('Filtered ECG')
        axarr[1].legend()
        f.savefig('/home/david/Report_filter/_lead_' + str(lead) + '_.png')
        plt.cla()
        plt.clf()
        plt.close()
    if i > 0:
        break
    print('Number of Patients treated for Experiment 1:', i + 1)