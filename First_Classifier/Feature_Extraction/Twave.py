import pandas as pd
from Preprocessing_features import compute_std, compute_median, compute_mean, maximum, minimum, bandpass_filter


def extraction_feature_Twave(ecg, freq, features_dict, lead, preprocess=False):
    Ttipo = features_dict['Ttipo']
    Ton_points = features_dict['Ton']
    Toff_points = features_dict['Toff']
    T_points = features_dict['T']
    T_locations = T_points[T_points > 0]
    Amplitudes = ecg[T_locations]
    indexes_T = Ton_points * Toff_points > 0
    Ton_locations, Toff_locations = Ton_points[indexes_T], Toff_points[indexes_T]
    durations = Toff_locations - Ton_locations
    Ttipo_freq_1 = len([value for value in Ttipo if value == 1]) / len(Ttipo)
    Ttipo_freq_2 = len([value for value in Ttipo if value == 2]) / len(Ttipo)
    Ttipo_freq_3 = len([value for value in Ttipo if value == 3]) / len(Ttipo)
    Ttipo_freq_4 = len([value for value in Ttipo if value == 4]) / len(Ttipo)
    Ttipo_freq_5 = len([value for value in Ttipo if value == 5]) / len(Ttipo)
    feat = pd.DataFrame({
        'Ttipo_mean_' + str(lead): [compute_mean(Ttipo)],
        'Ttipo_max_' + str(lead): [maximum(Ttipo)],
        'Ttipo_min_' + str(lead): [minimum(Ttipo)],
        'Ttipo_median_' + str(lead): [compute_median(Ttipo)],
        'Ttipo_std_' + str(lead): [compute_std(Ttipo)],
        'Ttipo_freq_1_' + str(lead): [Ttipo_freq_1],
        'Ttipo_freq_2_' + str(lead): [Ttipo_freq_2],
        'Ttipo_freq_3_' + str(lead): [Ttipo_freq_3],
        'Ttipo_freq_4_' + str(lead): [Ttipo_freq_4],
        'Ttipo_freq_5_' + str(lead): [Ttipo_freq_5],
        'Min_Tduration_' + str(lead): [minimum(durations)],
        'Max_Tduration_' + str(lead): [maximum(durations)],
        'Mean_Tduration_' + str(lead): [compute_mean(durations)],
        'Median_Tduration_' + str(lead): [compute_median(durations)],
        'Std_Tduration_' + str(lead): [compute_std(durations)],
        'Min_Tamp_' + str(lead): [minimum(Amplitudes)],
        'Max_Tamp_' + str(lead): [maximum(Amplitudes)],
        'Mean_Tamp_' + str(lead): [compute_mean(Amplitudes)],
        'Median_Tamp_' + str(lead): [compute_median(Amplitudes)],
        'Std_Tamp_' + str(lead): [compute_std(Amplitudes)]
    })
    return feat


