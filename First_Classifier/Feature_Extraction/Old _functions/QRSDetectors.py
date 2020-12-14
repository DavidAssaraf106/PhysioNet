import numpy
import wfdb
import pandas
import os
import numpy as np
from scipy.signal import butter, lfilter
from Preprocessing_features import bandpass_filter, adjusting_R_windows

# TODO: can we improve the recognition of the QRS complexes thanks to the paralleliztion of the work on several leads?

def QRS_detector_gqrs(ecg, num_lead):  # Detect the QRS Complex
    ecg_lead = ecg[num_lead]
    R_points = wfdb.processing.gqrs_detect(fs=500, sig=ecg_lead, adc_gain=1000, adc_zero=0)
    R_points = adjusting_R_windows(ecg[num_lead], R_points)
    return R_points

def QRS_detector_xqrs(ecg, num_lead):
    ecg_lead = ecg[num_lead]
    return adjusting_R_windows(ecg[num_lead], wfdb.processing.xqrs_detect(sig=ecg_lead, fs=500, verbose=False))

def QRS_detector_PT(ecg, num_lead):
    ecg_lead = ecg[num_lead]
    return adjusting_R_windows(ecg_lead, detect_qrs(ecg_measurements=ecg_lead, signal_frequency=500))

def detect_qrs(ecg, num_lead, signal_frequency):
    ecg_measurements = ecg[num_lead]
    refractory_period = 100
    qrs_peak_value = 0.0
    noise_peak_value = 0.0
    detected_peaks_values, detected_peaks_indices = detect_peaks(ecg_measurements=ecg_measurements, signal_frequency=signal_frequency)
    qrs_peaks_indices = np.array([], dtype=int)
    noise_peaks_indices = np.array([], dtype=int)
    qrs_peak_filtering_factor = 0.125
    noise_peak_filtering_factor = 0.125
    qrs_noise_diff_weight = 0.25
    threshold_value = 0.0
    for detected_peak_index, detected_peaks_value in zip(detected_peaks_indices, detected_peaks_values):
        try:
            last_qrs_index = qrs_peaks_indices[-1]
        except IndexError:
            last_qrs_index = 0
        if detected_peak_index - last_qrs_index > refractory_period or not qrs_peaks_indices.size:
            if detected_peaks_value > threshold_value:
                qrs_peaks_indices = np.append(qrs_peaks_indices, detected_peak_index)


                qrs_peak_value = qrs_peak_filtering_factor * detected_peaks_value + \
                                      (1 - qrs_peak_filtering_factor) * qrs_peak_value
            else:
                noise_peaks_indices = np.append(noise_peaks_indices, detected_peak_index)

                # Adjust noise peak value used later for setting QRS-noise threshold.
                noise_peak_value = noise_peak_filtering_factor * detected_peaks_value + \
                                        (1 - noise_peak_filtering_factor) * noise_peak_value

            # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
            threshold_value = noise_peak_value + \
                                   qrs_noise_diff_weight * (qrs_peak_value - noise_peak_value)

    return qrs_peaks_indices

def detect_peaks(ecg_measurements, signal_frequency):
    filter_lowcut = 0.001
    filter_highcut = 15.0
    filter_order = 1
    integration_window = 30
    findpeaks_limit = 0.35
    findpeaks_spacing = 100
    refractory_period = 100
    qrs_peak_filtering_factor = 0.125
    noise_peak_filtering_factor = 0.125
    qrs_noise_diff_weight = 0.25
    qrs_peaks_indices = np.array([], dtype=int)
    noise_peaks_indices = np.array([], dtype=int)
    filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut,
                                                signal_freq=signal_frequency, filter_order=filter_order)
    filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
    differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)
    squared_ecg_measurements = differentiated_ecg_measurements ** 2
    integrated_ecg_measurements = np.convolve(squared_ecg_measurements,
                                              np.ones(integration_window) / integration_window)
    detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                       limit=findpeaks_limit,
                                       spacing=findpeaks_spacing)

    detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

    return detected_peaks_values, detected_peaks_indices


def findpeaks(data, spacing=1, limit=None):

    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind