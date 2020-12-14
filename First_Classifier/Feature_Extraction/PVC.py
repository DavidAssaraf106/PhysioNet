from Preprocessing_features_Submission import bandpass_filter, compute_mean, compute_median, compute_std, minimum, maximum
import numpy as np
import matplotlib.mlab as mlab
import pandas as pd
import math


def comp_diff(R_points):  # @Jeremy, for PVC detection
    R_points = np.asarray(R_points)
    cnt_diff_ecg = []
    for idx_q in range(1, len(R_points)):
        cnt_diff = R_points[idx_q] - R_points[idx_q - 1]
        cnt_diff_ecg.append(cnt_diff)
    return cnt_diff_ecg


def extraction_feature_PVC(ecg, freq, features_dict, lead, preprocess=False):
    T = 3
    R_points = features_dict['R']
    R_locations = R_points[R_points > 0]
    feat = pd.DataFrame()
    if lead == 0:
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        Dqrs = Q_locations - S_locations
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        rr = comp_diff(R_locations)
        IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
        feat = pd.DataFrame({'Dqrsmax_' + str(lead): [maximum(Dqrs)],
                             'Dqrsmean_' + str(lead): [compute_mean(Dqrs)],
                             'Dqrsmed_' + str(lead): [compute_median(Dqrs)],
                             'Dqrsstd_' + str(lead): [compute_std(Dqrs)],
                             'IRmax_' + str(lead): [maximum(IR)],
                             'IRmedian_' + str(lead): [compute_median(IR)],
                             'IRmean_' + str(lead): [compute_mean(IR)],
                             'IRstd_' + str(lead): [compute_std(IR)],
                             })
    if lead == 1:
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        rr = comp_diff(R_locations)
        IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
        RR_index = R_points[:-1] * R_points[1:] > 0
        R_points_RR = R_points[:-1][RR_index]
        amplitudes = np.asarray(ecg[R_points_RR])
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        area = np.asarray([np.sum(ecg[Q_locations[i]:S_locations[i]]) for i in range(len(Q_locations))])
        feat = pd.DataFrame({'Fmax_' + str(lead): [maximum(amplitudes)],  # à l'ancienne il y avait que sur celui-ci
                             'Fmean_' + str(lead): [compute_mean(amplitudes)],
                             'Fmed_' + str(lead): [compute_median(amplitudes)],
                             'Fstd_' + str(lead): [compute_std(amplitudes)],
                             'Sqrsmax_' + str(lead): [maximum(area)],
                             'Sqrsmed_' + str(lead): [compute_median(area)],
                             'Sqrsmean_' + str(lead): [compute_mean(area)],
                             'Sqrsstd_' + str(lead): [compute_std(area)],
                             'IRmax_' + str(lead): [maximum(IR)],
                             'IRmedian_' + str(lead): [compute_median(IR)],
                             'IRmean_' + str(lead): [compute_mean(IR)],
                             'IRstd_' + str(lead): [compute_std(IR)]
                             })
    if lead == 2:
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        Q_locations_modif = Q_locations + T
        S_locations_modif = S_locations + T
        Xh = np.asarray(
            [np.asarray(ecg[Q_locations[i]:S_locations[i]]) ** 2 for i in range(len(Q_locations))])
        XhT = np.asarray(
            [np.asarray(ecg[Q_locations_modif[i]:S_locations_modif[i]]) ** 2 for i in range(len(Q_locations))])
        d = np.asarray([np.sqrt(Xh[i] + XhT[i]) for i in range(len(Xh))])
        d = [maximum(d[i] - minimum(d[i])) for i in range(len(d))]
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        rr = comp_diff(R_locations)
        IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
        feat = pd.DataFrame({'Amax_' + str(lead): [maximum(d)],
                             'Amedian_' + str(lead): [compute_median(d)],
                             'Amean_' + str(lead): [compute_mean(d)],
                             'Astd_' + str(lead): [compute_std(d)],
                             'IRmax_' + str(lead): [maximum(IR)],
                             'IRmedian_' + str(lead): [compute_median(IR)],
                             'IRmean_' + str(lead): [compute_mean(IR)],
                             'IRstd_' + str(lead): [compute_std(IR)]
                             })
    if lead == 3:
        R_points = features_dict['R']
        R_locations = R_points[R_points > 0]
        RR_index = R_points[:-1] * R_points[1:] > 0
        R_points_RR = R_points[:-1][RR_index]
        amplitudes = np.asarray(ecg[R_points_RR])
        DF = amplitudes[1:] - amplitudes[:-1]
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        area = np.asarray([np.sum(ecg[Q_locations[i]:S_locations[i]]) for i in range(len(Q_locations))])
        feat = pd.DataFrame({'DFmax_' + str(lead): [maximum(DF)],
                             'DFmean_' + str(lead): [compute_mean(DF)],
                             'DFmed_' + str(lead): [compute_median(DF)],
                             'DFstd_' + str(lead): [compute_std(DF)],
                             'Sqrsmax_' + str(lead): [maximum(area)],
                             'Sqrsmed_' + str(lead): [compute_median(area)],
                             'Sqrsmean_' + str(lead): [compute_mean(area)],
                             'Sqrssted_' + str(lead): [compute_std(area)],
                             'Slope_1_' + str(lead): power_spectrum_mlab(ecg, 500)[0],
                             'Slope_2_' + str(lead): power_spectrum_mlab(ecg, 500)[1]
                             })
    if lead == 5:
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        rr = comp_diff(R_locations)
        IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
        RR_index = R_points[:-1] * R_points[1:] > 0
        R_points_RR = R_points[:-1][RR_index]
        amplitudes = np.asarray(ecg[R_points_RR])
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        area = np.asarray([np.sum(ecg[Q_locations[i]:S_locations[i]]) for i in range(len(Q_locations))])
        feat = pd.DataFrame({'Fmax_' + str(lead): [maximum(amplitudes)],  # à l'ancienne il y avait que sur celui-ci
                             'Fmean_' + str(lead): [compute_mean(amplitudes)],
                             'Fmed_' + str(lead): [compute_median(amplitudes)],
                             'Fstd_' + str(lead): [compute_std(amplitudes)],
                             'Sqrsmax_' + str(lead): [maximum(area)],
                             'Sqrsmed_' + str(lead): [compute_median(area)],
                             'Sqrsmean_' + str(lead): [compute_mean(area)],
                             'Sqrssted_' + str(lead): [compute_std(area)],
                             'IRmax_' + str(lead): [maximum(IR)],
                             'IRmedian_' + str(lead): [compute_median(IR)],
                             'IRmean_' + str(lead): [compute_mean(IR)],
                             'IRstd_' + str(lead): [compute_std(IR)],
                             'Slope_1_' + str(lead): power_spectrum_mlab(ecg, 500)[0],
                             'Slope_2_' + str(lead): power_spectrum_mlab(ecg, 500)[1]
                             })
    if lead == 6:
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        R_points_RR = R_points[:-1][RR_index]
        amplitudes = np.asarray(ecg[R_points_RR])
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        Q_locations_modif = Q_locations + T
        S_locations_modif = S_locations + T
        Xh = np.asarray(
            [np.asarray(ecg[Q_locations[i]:S_locations[i]]) ** 2 for i in range(len(Q_locations))])
        XhT = np.asarray(
            [np.asarray(ecg[Q_locations_modif[i]:S_locations_modif[i]]) ** 2 for i in range(len(Q_locations))])
        d = np.asarray([np.sqrt(Xh[i] + XhT[i]) for i in range(len(Xh))])
        d = [maximum(d[i] - minimum(d[i])) for i in range(len(d))]
        R_points = features_dict['R']
        R_locations = R_points[R_points > 0]
        RR_index = R_points[:-1] * R_points[1:] > 0
        R_points_RR = R_points[:-1][RR_index]
        amplitudes = np.asarray(ecg[R_points_RR])
        DF = amplitudes[1:] - amplitudes[:-1]
        feat = pd.DataFrame({'Fmax_' + str(lead): [maximum(amplitudes)],  # à l'ancienne il y avait que sur celui-ci
                             'Fmean_' + str(lead): [compute_mean(amplitudes)],
                             'Fmed_' + str(lead): [compute_median(amplitudes)],
                             'Fstd_' + str(lead): [compute_std(amplitudes)],
                             'DFmax_' + str(lead): [maximum(DF)],
                             'DFmean_' + str(lead): [compute_mean(DF)],
                             'DFmed_' + str(lead): [compute_median(DF)],
                             'DFstd_' + str(lead): [compute_std(DF)],
                             'Amax_' + str(lead): [maximum(d)],
                             'Amedian_' + str(lead): [compute_median(d)],
                             'Amean_' + str(lead): [compute_mean(d)],
                             'Astd_' + str(lead): [compute_std(d)]})
    if lead == 7:
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        area = np.asarray([np.sum(ecg[Q_locations[i]:S_locations[i]]) for i in range(len(Q_locations))])
        feat = pd.DataFrame({'Sqrsmax_' + str(lead): [maximum(area)],
                             'Sqrsmed_' + str(lead): [compute_median(area)],
                             'Sqrsmean_' + str(lead): [compute_mean(area)],
                             'Sqrssted_' + str(lead): [compute_std(area)],
                             })
    if lead == 9:
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        R_points_RR = R_points[:-1][RR_index]
        amplitudes = np.asarray(ecg[R_points_RR])
        feat = pd.DataFrame({'Fmax_' + str(lead): [maximum(amplitudes)],  # à l'ancienne il y avait que sur celui-ci
                             'Fmean_' + str(lead): [compute_mean(amplitudes)],
                             'Fmed_' + str(lead): [compute_median(amplitudes)],
                             'Fstd_' + str(lead): [compute_std(amplitudes)]
                             })
    if lead == 10:
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        Dqrs = Q_locations - S_locations
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        rr = comp_diff(R_locations)
        IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
        feat = pd.DataFrame({'Dqrsmax_' + str(lead): [maximum(Dqrs)],
                             'Dqrsmean_' + str(lead): [compute_mean(Dqrs)],
                             'Dqrsmed_' + str(lead): [compute_median(Dqrs)],
                             'Dqrsstd_' + str(lead): [compute_std(Dqrs)],
                             'IRmax_' + str(lead): [maximum(IR)],
                             'IRmedian_' + str(lead): [compute_median(IR)],
                             'IRmean_' + str(lead): [compute_mean(IR)],
                             'IRstd_' + str(lead): [compute_std(IR)],
                             'Slope_1_' + str(lead): power_spectrum_mlab(ecg, 500)[0],
                             'Slope_2_' + str(lead): power_spectrum_mlab(ecg, 500)[1]
                             })
    if lead == 11:
        R_points = features_dict['R']
        RR_index = R_points[:-1] * R_points[1:] > 0
        rr = comp_diff(R_locations)
        IR = np.asarray(rr[RR_index]) / compute_mean(rr[RR_index])
        Q_locations = features_dict['Q']
        S_locations = features_dict['S']
        indexes_effectives = Q_locations * S_locations > 0
        Q_locations = Q_locations[indexes_effectives]
        S_locations = S_locations[indexes_effectives]
        Dqrs = Q_locations - S_locations
        feat = pd.DataFrame({'Dqrsmax_' + str(lead): [maximum(Dqrs)],
                             'Dqrsmean_' + str(lead): [compute_mean(Dqrs)],
                             'Dqrsmed_' + str(lead): [compute_median(Dqrs)],
                             'Dqrsstd_' + str(lead): [compute_std(Dqrs)],
                             'IRmax_' + str(lead): [maximum(IR)],
                             'IRmedian_' + str(lead): [compute_median(IR)],
                             'IRmean_' + str(lead): [compute_mean(IR)],
                             'IRstd_' + str(lead): [compute_std(IR)],
                             })
    return feat


def power_spectrum_mlab(ecg, freq):
    """
    This function implements the approach of: PVC discrimination using the QRS power spectrum and
    self-organizing maps M.L. Talbi ∗, A. Charef.
    The preprocessing expected in this paper is a bandpass filter (0.67-30) Hz.
    The features extracted are the slopes of the Power Spectrum in the log-domain.
    The power spectrum of the QRS is computed thanks to mlab approach (http://faculty.jsd.claremont.edu/jmilton/Math_Lab_tool/Labs/Lab9.pdf).
    We will use the loglog plot in order to visualize the graphs and then we will decide whether we compute slopes or not.
    In the paper, only the lead II is used, but we will do it in every lead.
    Input: lead of an ecg
    Output: Characteristic slopes of the power spectrum of the different leads: (12,2)-size array
    """
    ecg = bandpass_filter(ecg, 0.67, 30, 500, 3)
    n_fft = pow(2, math.ceil(math.log2(len(ecg))))
    pad_to = 0.1 * (n_fft - len(ecg))
    if pad_to == 0:
        pad_to = 100  # arbitrary
    power, freqs = mlab.psd(x=ecg, NFFT=32 * n_fft, Fs=int(freq), detrend='linear', noverlap=32,
                            pad_to=int(pad_to))
    freq_begin_window_one = np.argmin(np.abs(freqs - 3))
    if freq_begin_window_one == 0:
        freq_begin_window_one = 1
    freq_end_window_one = np.argmin(np.abs(freqs - 8))
    if freq_end_window_one > freq_begin_window_one + 1:
        coefficient_one = np.polyfit(np.log10(freqs[freq_begin_window_one: freq_end_window_one]),
                                     np.log10(power[freq_begin_window_one: freq_end_window_one]), 1)[0]
    else:
        coefficient_one = 0
    freq_begin_window_two = np.argmin(np.abs(freqs - 15))
    freq_end_window_two = np.argmin(np.abs(freqs - 19))
    if freq_end_window_two > freq_begin_window_two + 1:
        coefficient_two = np.polyfit(np.log10(freqs[freq_begin_window_two: freq_end_window_two]),
                                     np.log10(power[freq_begin_window_two: freq_end_window_two]), 1)[0]
    else:
        coefficient_two = 0
    return coefficient_one, coefficient_two



