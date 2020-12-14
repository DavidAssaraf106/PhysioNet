import numpy as np
import pandas as pd

from Preprocessing_features import *

# TODO: QRS duration to be computed as the mean over every lead: multi lead QRS boundaries

interesting_keys = ["Pon", "P", "Poff", "QRSon", "Q", "R", "S", "QRSoff", "Ton", "T", "Toff"]


def minimum(array):  # handles when the array is empty
    if len(array) > 0:
        return np.min(array)
    else:
        return 0


def compute_median(array):
    if len(array) > 0:
        return np.median(array)
    else:
        return 0


def maximum(array):
    if len(array) > 0:
        return np.max(array)
    else:
        return 0


def compute_std(array):
    if len(array) > 0:
        return np.std(array)
    else:
        return 0


def compute_mean(array):
    if len(array) > 0:
        return np.mean(array)
    else:
        return 0


# extraction_feature_I: PAC detection + AF detection
def extraction_feature_I(ecg, freq, features_dict):
    QRSOn_locations = features_dict['QRSon']
    R_points = features_dict['R']
    QRSOff_locations = features_dict['QRSoff']
    if len(R_points) > 5:

        rr = comp_diff(R_points)
        SD1, SD2 = comp_poincare(rr)
        interbeats = calculate_interbeat(R_points)

        indexes_QRS_effectives = np.asarray(QRSOn_locations) * np.asarray(QRSOff_locations) > 0
        QRSOn_locations_effective, QRSOff_locations_effective = QRSOn_locations[indexes_QRS_effectives], \
                                                                QRSOff_locations[
                                                                    indexes_QRS_effectives]
        width = calculate_QRS_Width(QRSOn_locations_effective, QRSOff_locations_effective)
        Area = calculate_QRS_Area(ecg, QRSOn_locations_effective, QRSOff_locations_effective)

        feat = pd.DataFrame({
            # Reference: AF Detection, Armand's repo.
            'cosEn_I': [comp_cosEn(rr)],
            'AFEv_I': [comp_AFEv(rr)],
            'OriginCount_I': [comp_OriginCount(rr)],
            'IrrEv_I': [comp_IrrEv(rr)],
            'PACEV_I': [comp_PACEv(rr)],

            'AVNN_I': [comp_AVNN(rr)],
            'SDNN_I': [comp_SDNN(rr)],
            'SEM_I': [comp_SEM(rr)],
            'minRR_I': [comp_minRR(rr)],
            'medHR_I': [comp_medHR(rr)],
            'PNN20_I': [comp_PNN20(rr)],
            'PNN50_I': [comp_PNN50(rr)],
            'RMSSD_I': [comp_RMSSD(rr)],
            'CV_I': [comp_CV(rr)],
            'SD1_I': [SD1],
            'SD2_I': [SD2],

            # PAC detection, the three features should have the same size so no worries about that

            'Interbeat_min_I': [minimum(interbeats)],
            'Interbeat_max_I': [maximum(interbeats)],
            'Interbeat_mean_I': [compute_mean(interbeats)],
            'Interbeat_median_I': [compute_median(interbeats)],
            'Interbeat_std_I': [compute_std(interbeats)],

            'QRS_Width_min_I': [minimum(width)],
            'QRS_Width_max_I': [maximum(width)],
            'QRS_Width_mean_I': [compute_mean(width)],
            'QRS_Width_median_I': [compute_median(width)],
            'QRS_Width_std_I': [compute_std(width)],

            'QRS_Area_min_I': [minimum(Area)],
            'QRS_Area_max_I': [maximum(Area)],
            'QRS_Area_mean_I': [compute_mean(Area)],
            'QRS_Area_median_I': [compute_median(Area)],
            'QRS_Area_std_I': [compute_std(Area)]

        })
    else:
        feat = pd.DataFrame({
            # Reference: AF Detection, Armand's repo.
            'cosEn_I': [0],
            'AFEv_I': [0],
            'OriginCount_I': [0],
            'IrrEv_I': [0],
            'PACEV_I': [0],

            'AVNN_I': [0],
            'SDNN_I': [0],
            'SEM_I': [0],
            'minRR_I': [0],
            'medHR_I': [0],
            'PNN20_I': [0],
            'PNN50_I': [0],
            'RMSSD_I': [0],
            'CV_I': [0],
            'SD1_I': [0],
            'SD2_I': [0],

            # PAC detection, the three features should have the same size so no worries about that

            'Interbeat_min_I': [0],
            'Interbeat_max_I': [0],
            'Interbeat_mean_I': [0],
            'Interbeat_median_I': [0],
            'Interbeat_std_I': [0],

            'QRS_Width_min_I': [0],
            'QRS_Width_max_I': [0],
            'QRS_Width_mean_I': [0],
            'QRS_Width_median_I': [0],
            'QRS_Width_std_I': [0],

            'QRS_Area_min_I': [0],
            'QRS_Area_max_I': [0],
            'QRS_Area_mean_I': [0],
            'QRS_Area_median_I': [0],
            'QRS_Area_std_I': [0]

        })
    # print(feat)
    return feat


# extraction_feature_II: only useful for detection of PVC + PAC + AF, based on the Energy notion.
def extraction_feature_II(ecg, freq, features_dict):

    QRSOn_locations = features_dict['QRSon']
    Q_points = features_dict['Q']
    R_points = features_dict['R']
    S_points = features_dict['S']
    QRSOff_locations = features_dict['QRSoff']
    if len(R_points) > 5:
        rr = comp_diff(R_points)
        SD1, SD2 = comp_poincare(rr)

        indexes_QRS_effectives = QRSOn_locations * QRSOff_locations > 0
        QRSOn_locations_effective, QRSOff_locations_effective = QRSOn_locations[indexes_QRS_effectives], \
                                                                QRSOff_locations[
                                                                    indexes_QRS_effectives]
        interbeat = calculate_interbeat(R_points)
        width = calculate_QRS_Width(QRSOn_locations_effective, QRSOff_locations_effective)
        area = calculate_QRS_Area(ecg, QRSOn_locations_effective, QRSOff_locations_effective)
        feat = pd.DataFrame({
            # Reference: AF Detection, Armand's repo.
            'cosEn_II': [comp_cosEn(rr)],
            'AFEv_II': [comp_AFEv(rr)],
            'OriginCount_II': [comp_OriginCount(rr)],
            'IrrEv_II': [comp_IrrEv(rr)],
            'PACEV_II': [comp_PACEv(rr)],

            'AVNN_II': [comp_AVNN(rr)],
            'SDNN_II': [comp_SDNN(rr)],
            'SEM_II': [comp_SEM(rr)],
            'minRR_II': [comp_minRR(rr)],
            'medHR_II': [comp_medHR(rr)],
            'PNN20_II': [comp_PNN20(rr)],
            'PNN50_II': [comp_PNN50(rr)],
            'RMSSD_II': [comp_RMSSD(rr)],
            'CV_II': [comp_CV(rr)],
            'SD1_II': [SD1],
            'SD2_II': [SD2],

            # Reference: Cardiac Arrhythmia Detection By ECG Feature Extraction
            # 'Sample_RR': [sample_RR(ecg, R_points)],

            'ecg_energy_II': [calculate_energy(ecg)],
            'average_energy_peak_II': [compute_mean(energy_peak(ecg, Q_points, S_points))],
            'median_energy_peak_II': [compute_median(energy_peak(ecg, Q_points, S_points))],

            # Reference: Premature Ventricular Contraction (PVC) Detection Using R Signals
            'average_time_difference_II': [compute_mean(comp_diff(R_points))],
            'median_time_difference_II': [compute_median(comp_diff(R_points))],

            # Reference: Bayesian Classification Models for Premature Ventricular Contraction Detection on ECG Traces
            'min_ECG_II': [np.min(ecg)],
            'max_ECG_II': [np.max(ecg)],
            'mean_ECG_II': [compute_mean(ecg)],
            'std_ECG_II': [compute_std(ecg)],

            # PAC Detection
            'Interbeat_min_II': [minimum(interbeat)],
            'Interbeat_max_II': [maximum(interbeat)],
            'Interbeat_mean_II': [compute_mean(interbeat)],
            'Interbeat_median_II': [compute_median(interbeat)],
            'Interbeat_std_II': [compute_std(interbeat)],

            'QRS_Width_min_II': [minimum(width)],
            'QRS_Width_max_II': [maximum(width)],
            'QRS_Width_mean_II': [compute_mean(width)],
            'QRS_Width_median_II': [compute_median(width)],
            'QRS_Width_std_II': [compute_std(width)],

            'QRS_Area_min_II': [minimum(area)],
            'QRS_Area_max_II': [maximum(area)],
            'QRS_Area_mean_II': [compute_mean(area)],
            'QRS_Area_median_II': [compute_median(area)],
            'QRS_Area_std_II': [compute_std(area)]
        })
    else:
        feat = pd.DataFrame({
            # Reference: AF Detection, Armand's repo.
            'cosEn_II': [0],
            'AFEv_II': [0],
            'OriginCount_II': [0],
            'IrrEv_II': [0],
            'PACEV_II': [0],

            'AVNN_II': [0],
            'SDNN_II': [0],
            'SEM_II': [0],
            'minRR_II': [0],
            'medHR_II': [0],
            'PNN20_II': [0],
            'PNN50_II': [0],
            'RMSSD_II': [0],
            'CV_II': [0],
            'SD1_II': [0],
            'SD2_II': [0],

            # Reference: Cardiac Arrhythmia Detection By ECG Feature Extraction
            # 'Sample_RR': [sample_RR(ecg, R_points)],

            'ecg_energy_II': [0],
            'average_energy_peak_II': [0],
            'median_energy_peak_II': [0],

            # Reference: Premature Ventricular Contraction (PVC) Detection Using R Signals
            'average_time_difference_II': [0],
            'median_time_difference_II': [0],

            # Reference: Bayesian Classification Models for Premature Ventricular Contraction Detection on ECG Traces
            'min_ECG_II': [np.min(ecg)],
            'max_ECG_II': [np.max(ecg)],
            'mean_ECG_II': [compute_mean(ecg)],
            'std_ECG_II': [compute_std(ecg)],

            # PAC Detection
            'Interbeat_min_II': [0],
            'Interbeat_max_II': [0],
            'Interbeat_mean_II': [0],
            'Interbeat_median_II': [0],
            'Interbeat_std_II': [0],

            'QRS_Width_min_II': [0],
            'QRS_Width_max_II': [0],
            'QRS_Width_mean_II': [0],
            'QRS_Width_median_II': [0],
            'QRS_Width_std_II': [0],

            'QRS_Area_min_II': [0],
            'QRS_Area_max_II': [0],
            'QRS_Area_mean_II': [0],
            'QRS_Area_median_II': [0],
            'QRS_Area_std_II': [0]

        })
    return feat


# extraction_feature_V1: useful for LBBB and RBBB, and I-AVB
def extraction_feature_V1(ecg, freq, features_dict):
    QRSOn_locations = features_dict['QRSon']
    Q_points = features_dict['Q']
    R_points = features_dict['R']
    S_points = features_dict['S']
    QRSOff_locations = features_dict['QRSoff']
    POn_locations = features_dict['Pon']
    T_locations = features_dict['T']
    TOff_locations = features_dict['Toff']
    if len(R_points) > 5:
        T_peaks_effective = T_locations[T_locations > 0]
        indexes_QRS_time = QRSOn_locations * QRSOff_locations > 0  # the indexes where we will be able to compute the QRS duration
        QRSOn_locations_effectives_time, QRSOff_locations_effectives_time = QRSOn_locations[indexes_QRS_time], \
                                                                            QRSOff_locations[indexes_QRS_time]

        indexes_QRSOn_effectives = QRSOn_locations > 0
        Q_points = Q_points[:len(indexes_QRSOn_effectives)]
        R_points = R_points[:len(indexes_QRSOn_effectives)]
        S_points = S_points[:len(indexes_QRSOn_effectives)]

        QRSOn_locations_effectives, R_points_effective, S_points_effective, Q_points_effective = \
            QRSOn_locations[indexes_QRSOn_effectives], R_points[indexes_QRSOn_effectives[:len(R_points)]], \
            S_points[indexes_QRSOn_effectives[:len(R_points)]], Q_points[indexes_QRSOn_effectives[:len(R_points)]]
        indexes_QT_interval = QRSOn_locations * TOff_locations > 0
        QRSOn_locations_effectives_QT, TOff_locations_effectives_QT = QRSOn_locations[indexes_QT_interval], \
                                                                      TOff_locations[
                                                                          indexes_QT_interval]

        indexes_PR_interval = POn_locations * QRSOn_locations > 0
        QRSOn_locations_effectives_PR, POn_locations_effectives_PR = QRSOn_locations[indexes_PR_interval], \
                                                                     POn_locations[
                                                                         indexes_PR_interval]

        S_duration = calculate_S_duration(ecg, R_points_effective, S_points_effective, QRSOn_locations_effectives)
        R_duration = calculate_R_duration(ecg, Q_points_effective, R_points_effective, QRSOn_locations_effectives)
        QRS_time = QRSOff_locations_effectives_time - QRSOn_locations_effectives_time
        QT_interval = TOff_locations_effectives_QT - QRSOn_locations_effectives_QT
        PR_interval = QRSOn_locations_effectives_PR - POn_locations_effectives_PR

        feat = pd.DataFrame({

            # Reference: Cardiac Arrhythmia Detection By ECG Feature Extraction
            'T_wave_V1': [compute_mean(ecg[T_peaks_effective])],
            'T_wave inversion_V1': [int(compute_mean(ecg[T_peaks_effective]) < 0)],

            'Duration_S_wave_min_V1': [minimum(S_duration)],
            'Duration_S_wave_max_V1': [maximum(S_duration)],
            'Duration_S_wave_mean_V1': [compute_mean(S_duration)],
            'Duration_S_wave_median_V1': [compute_median(S_duration)],
            'Duration_S_wave_std_V1': [compute_std(S_duration)],

            'Duration_R_wave_min_V1': [minimum(R_duration)],
            'Duration_R_wave_max_V1': [maximum(R_duration)],
            'Duration_R_wave_mean_V1': [compute_mean(R_duration)],
            'Duration_R_wave_median_V1': [compute_median(R_duration)],
            'Duration_R_wave_std_V1': [compute_std(R_duration)],
            'Dominant_RWave_Coefficient_V1': [calculate_coef_Rwave(ecg, Q_points, S_points)],

            'average_RQ_interval_V1': [compute_mean(R_points - Q_points)],
            'average_SR_interval_V1': [compute_mean(S_points - R_points)],
            'median_RQ_interval_V1': [compute_median(R_points - Q_points)],
            'median_SR_interval_V1': [compute_median(S_points - R_points)],
            'min_RQ_interval_V1': [minimum(R_points - Q_points)],
            'min_SR_interval_V1': [minimum(S_points - R_points)],

            'average_num_zero_crossing_V1': [compute_mean(zero_crossing(ecg, Q_points, S_points))],
            'median_num_zero_crossing_V1': [compute_median(zero_crossing(ecg, Q_points, S_points))],

            'median_QRS_time_V1': [compute_mean(QRS_time)],
            'min_QRS_time_V1': [minimum(QRS_time)],
            'mean_QRS_time_V1': [compute_mean(QRS_time)],
            'max_QRS_time_V1': [maximum(QRS_time)],

            # I-AVB detection:

            'median_QT_interval_V1': [compute_median(QT_interval)],
            'average_QT_interval_V1': [compute_mean(QT_interval)],
            'max_QT_interval_V1': [maximum(QT_interval)],
            'min_QT_interval_V1': [minimum(QT_interval)],

            'median_PR_interval_V1': [compute_median(PR_interval)],
            'average_PR_interval_V1': [compute_mean(PR_interval)],
            'max_PR_interval_V1': [maximum(PR_interval)],
            'min_PR_interval_V1': [minimum(PR_interval)]

        })
    else:
        feat = pd.DataFrame({

            # Reference: Cardiac Arrhythmia Detection By ECG Feature Extraction
            'T_wave_V1': [0],
            'T_wave inversion_V1': [0],

            'Duration_S_wave_min_V1': [0],
            'Duration_S_wave_max_V1': [0],
            'Duration_S_wave_mean_V1': [0],
            'Duration_S_wave_median_V1': [0],
            'Duration_S_wave_std_V1': [0],

            'Duration_R_wave_min_V1': [0],
            'Duration_R_wave_max_V1': [0],
            'Duration_R_wave_mean_V1': [0],
            'Duration_R_wave_median_V1': [0],
            'Duration_R_wave_std_V1': [0],
            'Dominant_RWave_Coefficient_V1': [0],

            'average_RQ_interval_V1': [0],
            'average_SR_interval_V1': [0],
            'median_RQ_interval_V1': [0],
            'median_SR_interval_V1': [0],
            'min_RQ_interval_V1': [0],
            'min_SR_interval_V1': [0],

            'average_num_zero_crossing_V1': [0],
            'median_num_zero_crossing_V1': [0],

            'median_QRS_time_V1': [0],
            'min_QRS_time_V1': [0],
            'mean_QRS_time_V1': [0],
            'max_QRS_time_V1': [0],

            # I-AVB detection:

            'median_QT_interval_V1': [0],
            'average_QT_interval_V1': [0],
            'max_QT_interval_V1': [0],
            'min_QT_interval_V1': [0],

            'median_PR_interval_V1': [0],
            'average_PR_interval_V1': [0],
            'max_PR_interval_V1': [0],
            'min_PR_interval_V1': [0]

        })
    # print(feat)
    return feat


def extraction_feature_V2(ecg, freq, features_dict):
    QRSOn_locations = features_dict['QRSon']
    R_points = features_dict['R']
    QRSOff_locations = features_dict['QRSoff']
    POn_locations = features_dict['Pon']
    POff_locations = features_dict['Poff']
    TOff_locations = features_dict['Toff']
    if len(R_points) > 5:
        indexes_relevant_1 = POff_locations * QRSOn_locations > 0
        POff_locations_effective, QRSOn_locations_effective = POff_locations[indexes_relevant_1], QRSOn_locations[
            indexes_relevant_1]
        indexes_relevant_2 = TOff_locations * POn_locations > 0
        TOff_locations_effective, POn_locations_effective = TOff_locations[indexes_relevant_2], POn_locations[
            indexes_relevant_2]
        Baseline_1 = Baseline(ecg, TOff_locations_effective, POn_locations_effective)
        Baseline_2 = Baseline(ecg, POff_locations_effective, QRSOn_locations_effective)
        setoff1 = compute_setoff(ecg, QRSOff_locations, freq, 0.01)
        setoff2 = compute_setoff(ecg, QRSOff_locations, freq, 0.04)
        setoff3 = compute_setoff(ecg, QRSOff_locations, freq, 0.08)

        feat = pd.DataFrame({

            'Isoelectric_line_1_max_V2': [maximum(Baseline_1)],
            'Isoelectric_line_1_min_V2': [minimum(Baseline_1)],
            'Isoelectric_line_1_mean_V2': [compute_mean(Baseline_1)],
            'Isoelectric_line_1_median_V2': [compute_median(Baseline_1)],
            'Isoelectric_line_1_std_V2': [compute_std(Baseline_1)],

            'Isoelectric_line_2_max_V2': [maximum(Baseline_2)],
            'Isoelectric_line_2_min_V2': [minimum(Baseline_2)],
            'Isoelectric_line_2_mean_V2': [compute_mean(Baseline_2)],
            'Isoelectric_line_2_median_V2': [compute_median(Baseline_2)],
            'Isoelectric_line_2_std_V2': [compute_std(Baseline_2)],

            'ST_elevation_1_max_V2': [maximum(ecg[QRSOff_locations])],
            'ST_elevation_1_min_V2': [minimum(ecg[QRSOff_locations])],
            'ST_elevation_1_mean_V2': [compute_mean(ecg[QRSOff_locations])],
            'ST_elevation_1_median_V2': [compute_median(ecg[QRSOff_locations])],
            'ST_elevation_1_std_V2': [compute_std(ecg[QRSOff_locations])],

            'ST_elevation_2_V2_max': [setoff1[0]],
            'ST_elevation_2_V2_min': [setoff1[1]],
            'ST_elevation_2_V2_mean': [setoff1[2]],
            'ST_elevation_2_V2_median': [setoff1[3]],
            'ST_elevation_2_V2_std': [setoff1[4]],
            'ST_elevation_3_V2_max': [setoff2[0]],
            'ST_elevation_3_V2_min': [setoff2[1]],
            'ST_elevation_3_V2_mean': [setoff2[2]],
            'ST_elevation_3_V2_median': [setoff2[3]],
            'ST_elevation_3_V2_std': [setoff2[4]],
            'ST_elevation_4_V2_max': [setoff3[0]],
            'ST_elevation_4_V2_min': [setoff3[1]],
            'ST_elevation_4_V2_mean': [setoff3[2]],
            'ST_elevation_4_V2_median': [setoff3[3]],
            'ST_elevation_4_V2_std': [setoff3[4]]
        })
    else:
        feat = pd.DataFrame({

            'Isoelectric_line_1_max_V2': [0],
            'Isoelectric_line_1_min_V2': [0],
            'Isoelectric_line_1_mean_V2': [0],
            'Isoelectric_line_1_median_V2': [0],
            'Isoelectric_line_1_std_V2': [0],

            'Isoelectric_line_2_max_V2': [0],
            'Isoelectric_line_2_min_V2': [0],
            'Isoelectric_line_2_mean_V2': [0],
            'Isoelectric_line_2_median_V2': [0],
            'Isoelectric_line_2_std_V2': [0],

            'ST_elevation_1_max_V2': [0],
            'ST_elevation_1_min_V2': [0],
            'ST_elevation_1_mean_V2': [0],
            'ST_elevation_1_median_V2': [0],
            'ST_elevation_1_std_V2': [0],

            'ST_elevation_2_V2_max': [0],
            'ST_elevation_2_V2_min': [0],
            'ST_elevation_2_V2_mean': [0],
            'ST_elevation_2_V2_median': [0],
            'ST_elevation_2_V2_std': [0],
            'ST_elevation_3_V2_max': [0],
            'ST_elevation_3_V2_min': [0],
            'ST_elevation_3_V2_mean': [0],
            'ST_elevation_3_V2_median': [0],
            'ST_elevation_3_V2_std': [0],
            'ST_elevation_4_V2_max': [0],
            'ST_elevation_4_V2_min': [0],
            'ST_elevation_4_V2_mean': [0],
            'ST_elevation_4_V2_median': [0],
            'ST_elevation_4_V2_std': [0]
        })
    # print(feat)
    return feat


def extraction_feature_V3(ecg, freq, features_dict):
    QRSOn_locations = features_dict['QRSon']
    R_points = features_dict['R']
    QRSOff_locations = features_dict['QRSoff']
    POn_locations = features_dict['Pon']
    POff_locations = features_dict['Poff']
    TOff_locations = features_dict['Toff']
    if len(R_points) > 5:
        indexes_relevant_1 = POff_locations * QRSOn_locations > 0
        POff_locations_effective, QRSOn_locations_effective = POff_locations[indexes_relevant_1], QRSOn_locations[
            indexes_relevant_1]
        indexes_relevant_2 = TOff_locations * POn_locations > 0
        TOff_locations_effective, POn_locations_effective = TOff_locations[indexes_relevant_2], POn_locations[
            indexes_relevant_2]
        Baseline_1 = Baseline(ecg, TOff_locations_effective, POff_locations_effective)
        Baseline_2 = Baseline(ecg, POff_locations_effective, QRSOn_locations_effective)
        setoff1 = compute_setoff(ecg, QRSOff_locations, freq, 0.01)
        setoff2 = compute_setoff(ecg, QRSOff_locations, freq, 0.04)
        setoff3 = compute_setoff(ecg, QRSOff_locations, freq, 0.08)

        feat = pd.DataFrame({

            'Isoelectric_line_1_max_V3': [maximum(Baseline_1)],
            'Isoelectric_line_1_min_V3': [minimum(Baseline_1)],
            'Isoelectric_line_1_mean_V3': [compute_mean(Baseline_1)],
            'Isoelectric_line_1_median_V3': [compute_median(Baseline_1)],
            'Isoelectric_line_1_std_V3': [compute_std(Baseline_1)],

            'Isoelectric_line_2_max_V3': [maximum(Baseline_2)],
            'Isoelectric_line_2_min_V3': [minimum(Baseline_2)],
            'Isoelectric_line_2_mean_V3': [compute_mean(Baseline_2)],
            'Isoelectric_line_2_median_V3': [compute_median(Baseline_2)],
            'Isoelectric_line_2_std_V3': [compute_std(Baseline_2)],

            'ST_elevation_1_max_V3': [maximum(ecg[QRSOff_locations])],
            'ST_elevation_1_min_V3': [minimum(ecg[QRSOff_locations])],
            'ST_elevation_1_mean_V3': [compute_mean(ecg[QRSOff_locations])],
            'ST_elevation_1_median_V3': [compute_median(ecg[QRSOff_locations])],
            'ST_elevation_1_std_V3': [compute_std(ecg[QRSOff_locations])],

            'ST_elevation_2_V3_max': [setoff1[0]],
            'ST_elevation_2_V3_min': [setoff1[1]],
            'ST_elevation_2_V3_mean': [setoff1[2]],
            'ST_elevation_2_V3_median': [setoff1[3]],
            'ST_elevation_2_V3_std': [setoff1[4]],
            'ST_elevation_3_V3_max': [setoff2[0]],
            'ST_elevation_3_V3_min': [setoff2[1]],
            'ST_elevation_3_V3_mean': [setoff2[2]],
            'ST_elevation_3_V3_median': [setoff2[3]],
            'ST_elevation_3_V3_std': [setoff2[4]],
            'ST_elevation_4_V3_max': [setoff3[0]],
            'ST_elevation_4_V3_min': [setoff3[1]],
            'ST_elevation_4_V3_mean': [setoff3[2]],
            'ST_elevation_4_V3_median': [setoff3[3]],
            'ST_elevation_4_V3_std': [setoff3[4]]
        })
    else:
        feat = pd.DataFrame({

            'Isoelectric_line_1_max_V3': [0],
            'Isoelectric_line_1_min_V3': [0],
            'Isoelectric_line_1_mean_V3': [0],
            'Isoelectric_line_1_median_V3': [0],
            'Isoelectric_line_1_std_V3': [0],

            'Isoelectric_line_2_max_V3': [0],
            'Isoelectric_line_2_min_V3': [0],
            'Isoelectric_line_2_mean_V3': [0],
            'Isoelectric_line_2_median_V3': [0],
            'Isoelectric_line_2_std_V3': [0],

            'ST_elevation_1_max_V3': [0],
            'ST_elevation_1_min_V3': [0],
            'ST_elevation_1_mean_V3': [0],
            'ST_elevation_1_median_V3': [0],
            'ST_elevation_1_std_V3': [0],

            'ST_elevation_2_V3_max': [0],
            'ST_elevation_2_V3_min': [0],
            'ST_elevation_2_V3_mean': [0],
            'ST_elevation_2_V3_median': [0],
            'ST_elevation_2_V3_std': [0],
            'ST_elevation_3_V3_max': [0],
            'ST_elevation_3_V3_min': [0],
            'ST_elevation_3_V3_mean': [0],
            'ST_elevation_3_V3_median': [0],
            'ST_elevation_3_V3_std': [0],
            'ST_elevation_4_V3_max': [0],
            'ST_elevation_4_V3_min': [0],
            'ST_elevation_4_V3_mean': [0],
            'ST_elevation_4_V3_median': [0],
            'ST_elevation_4_V3_std': [0]
        })

    # print(feat)
    return feat


# lead V6 is useful for LBBB, RBBB and I-AVB
def extraction_feature_V6(ecg, freq, features_dict):
    QRSOn_locations = features_dict['QRSon']
    Q_points = features_dict['Q']
    R_points = features_dict['R']
    S_points = features_dict['S']
    QRSOff_locations = features_dict['QRSoff']
    POn_locations = features_dict['Pon']
    T_locations = features_dict['T']
    TOff_locations = features_dict['Toff']
    if len(R_points) > 5:
        T_peaks_effective = T_locations[T_locations > 0]

        indexes_QRS_time = QRSOn_locations * QRSOff_locations > 0  # the indexes where we will be able to compute the QRS duration
        QRSOn_locations_effectives, QRSOff_locations_effectives = QRSOn_locations[indexes_QRS_time], QRSOff_locations[
            indexes_QRS_time]

        indexes_QT_interval = QRSOn_locations * TOff_locations > 0
        QRSOn_locations_effectives_QT, TOff_locations_effectives_QT = QRSOn_locations[indexes_QT_interval], \
                                                                      TOff_locations[
                                                                          indexes_QT_interval]

        indexes_PR_interval = POn_locations * QRSOn_locations > 0
        QRSOn_locations_effectives_PR, POn_locations_effectives_PR = QRSOn_locations[indexes_PR_interval], \
                                                                     POn_locations[
                                                                         indexes_PR_interval]
        QRS_time = QRSOff_locations_effectives - QRSOn_locations_effectives
        QT_interval = TOff_locations_effectives_QT - QRSOn_locations_effectives_QT
        PR_interval = QRSOn_locations_effectives_PR - POn_locations_effectives_PR

        feat = pd.DataFrame({

            # Reference: Cardiac Arrhythmia Detection By ECG Feature Extraction
            'Dominant_RWave_Coefficient_V6': [calculate_coef_Rwave(ecg, Q_points, S_points)],
            'T_wave_V6': [compute_mean(ecg[T_peaks_effective])],
            'T_wave inversion_V6': [int(compute_mean(ecg[T_peaks_effective]) < 0)],

            'median_QRS_time_V6': [compute_mean(QRS_time)],
            'min_QRS_time_V6': [minimum(QRS_time)],
            'average_QRS_time_V6': [compute_mean(QRS_time)],
            'max_QRS_time_V6': [maximum(QRS_time)],

            # I-AVB detection:
            'median_QT_interval_V6': [compute_median(QT_interval)],
            'average_QT_interval_V6': [compute_mean(QT_interval)],
            'max_QT_interval_V6': [maximum(QT_interval)],
            'min_QT_interval_V6': [minimum(QT_interval)],

            'median_PR_interval_V6': [compute_median(PR_interval)],
            'average_PR_interval_V6': [compute_mean(PR_interval)],
            'max_PR_interval_V6': [maximum(PR_interval)],
            'min_PR_interval_V6': [minimum(PR_interval)]
        })
    else:
        feat = pd.DataFrame({

            # Reference: Cardiac Arrhythmia Detection By ECG Feature Extraction
            'Dominant_RWave_Coefficient_V6': [0],
            'T_wave_V6': [0],
            'T_wave inversion_V6': [0],

            'median_QRS_time_V6': [0],
            'min_QRS_time_V6': [0],
            'average_QRS_time_V6': [0],
            'max_QRS_time_V6': [0],

            # I-AVB detection:
            'median_QT_interval_V6': [0],
            'average_QT_interval_V6': [0],
            'max_QT_interval_V6': [0],
            'min_QT_interval_V6': [0],

            'median_PR_interval_V6': [0],
            'average_PR_interval_V6': [0],
            'max_PR_interval_V6': [0],
            'min_PR_interval_V6': [0]
        })

    # print(feat)
    return feat


def extraction_feature_V4(ecg, freq, Q_points, R_points, S_points, POn_locations, P_locations, POff_locations,
                          QRSOn_locations, QRSOff_locations, TOn_locations, T_locations, TOff_locations):
    feat = pd.DataFrame({})
    return feat


def extraction_feature_V5(ecg, freq, Q_points, R_points, S_points, POn_locations, P_locations, POff_locations,
                          QRSOn_locations, QRSOff_locations, TOn_locations, T_locations, TOff_locations):
    feat = pd.DataFrame({})
    return feat


def extraction_feature_III(ecg, freq, Q_points, R_points, S_points, POn_locations, P_locations,
                           POff_locations, QRSOn_locations, QRSOff_locations, TOn_locations, T_locations,
                           TOff_locations):
    feat = pd.DataFrame({})
    return feat


def extraction_feature_aVR(ecg, freq, Q_points, R_points, S_points, POn_locations, P_locations,
                           POff_locations, QRSOn_locations, QRSOff_locations, TOn_locations, T_locations,
                           TOff_locations):  # add T_point, QRSOn, QRSOff
    feat = pd.DataFrame({})
    return feat


def extraction_feature_aVL(ecg, freq, Q_points, R_points, S_points, POn_locations, P_locations,
                           POff_locations, QRSOn_locations, QRSOff_locations, TOn_locations, T_locations,
                           TOff_locations):
    feat = pd.DataFrame({})
    return feat


def extraction_feature_aVF(ecg, freq, Q_points, R_points, S_points, POn_locations, P_locations,
                           POff_locations, QRSOn_locations, QRSOff_locations, TOn_locations, T_locations,
                           TOff_locations):
    feat = pd.DataFrame({})
    return feat
