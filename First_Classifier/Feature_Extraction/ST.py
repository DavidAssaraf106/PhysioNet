import numpy as np
import pandas as pd
from Preprocessing_features_Submission import compute_mean, compute_std, compute_median, maximum


def extraction_feature_ST(ecg, freq, features_dict, lead, preprocess=False):
    feat = pd.DataFrame()
    if lead == 0:
        QRSoff = features_dict['QRSoff']
        Ton = features_dict['Ton']
        indexes_ST = QRSoff * Ton > 0
        longueur_ref = min(len(QRSoff), len(Ton))
        QRSoff_here = QRSoff[:longueur_ref]
        Ton_here = Ton[:longueur_ref]
        indexes_ST_here = indexes_ST[:longueur_ref]
        QRSoff_ST, Ton = QRSoff_here[indexes_ST_here], Ton_here[indexes_ST_here]
        ST_segments = [ecg[QRSoff_ST[i]:Ton[i]] for i in range(len(QRSoff_ST))]
        ST_segments_mean = compute_mean([compute_mean(ST_segments[i]) for i in range(len(ST_segments))])
        ST_segments_median = compute_median([compute_median(ST_segments[i]) for i in range(len(ST_segments))])
        ST_segments_std = compute_std([compute_std(ST_segments[i]) for i in range(len(ST_segments))])
        feat = pd.DataFrame({'ST_segment_mean_' + str(lead): [ST_segments_mean],
                             'ST_segment_median_' + str(lead): [ST_segments_median],
                             'ST_segment_std_' + str(lead): [ST_segments_std]})
    if lead == 2:
        QRSoff = features_dict['QRSoff']
        Ton = features_dict['Ton']
        indexes_ST = QRSoff * Ton > 0
        longueur_ref = min(len(QRSoff), len(Ton))
        QRSoff_here = QRSoff[:longueur_ref]
        Ton_here = Ton[:longueur_ref]
        indexes_ST_here = indexes_ST[:longueur_ref]
        QRSoff_ST, Ton = QRSoff_here[indexes_ST_here], Ton_here[indexes_ST_here]
        ST_segments = [ecg[QRSoff_ST[i]:Ton[i]] for i in range(len(QRSoff_ST))]
        ST_segments_mean = compute_mean([compute_mean(ST_segments[i]) for i in range(len(ST_segments))])
        ST_segments_median = compute_median([compute_median(ST_segments[i]) for i in range(len(ST_segments))])
        ST_segments_std = compute_std([compute_std(ST_segments[i]) for i in range(len(ST_segments))])
        QRSon = features_dict['QRSon']
        QRSoff = features_dict['QRSoff']
        indexes_effectives_QRS = QRSon * QRSoff > 0
        QRSon_duration, QRSoff_duration = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        amplitudes_2 = ecg[np.asarray(QRSoff_duration) + 40]
        feat = pd.DataFrame({'amplitudes_2_max_' + str(lead): [maximum(amplitudes_2)],
                             'amplitudes_2_mean_' + str(lead): [compute_mean(amplitudes_2)],
                             'amplitudes_2_median_' + str(lead): [compute_median(amplitudes_2)],
                             'amplitudes_2_std_' + str(lead): [compute_std(amplitudes_2)],
                             'ST_segment_mean_' + str(lead): [ST_segments_mean],
                             'ST_segment_median_' + str(lead): [ST_segments_median],
                             'ST_segment_std_' + str(lead): [ST_segments_std]
                             })
    if lead == 3:
        QRSon = features_dict['QRSon']
        QRSoff = features_dict['QRSoff']
        indexes_effectives_QRS = QRSon * QRSoff > 0
        QRSon_duration, QRSoff_duration = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        amplitudes_0 = ecg[QRSon_duration]
        amplitudes_2 = ecg[np.asarray(QRSoff_duration) + 40]
        feat = pd.DataFrame({'amplitudes_0_max_' + str(lead): [maximum(amplitudes_0)],
                             'amplitudes_0_mean_' + str(lead): [compute_mean(amplitudes_0)],
                             'amplitudes_0_median_' + str(lead): [compute_median(amplitudes_0)],
                             'amplitudes_0_std_' + str(lead): [compute_std(amplitudes_0)],
                             'amplitudes_2_max_' + str(lead): [maximum(amplitudes_2)],
                             'amplitudes_2_mean_' + str(lead): [compute_mean(amplitudes_2)],
                             'amplitudes_2_median_' + str(lead): [compute_median(amplitudes_2)],
                             'amplitudes_2_std_' + str(lead): [compute_std(amplitudes_2)]
                             })
    if lead == 5:
        QRSon = features_dict['QRSon']
        QRSoff = features_dict['QRSoff']
        indexes_effectives_QRS = QRSon * QRSoff > 0
        QRSon_duration, QRSoff_duration = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        amplitudes_1 = ecg[np.asarray(QRSoff_duration) + 30]
        feat = pd.DataFrame({'amplitudes_1_max_' + str(lead): [maximum(amplitudes_1)],
                             'amplitudes_1_mean_' + str(lead): [compute_mean(amplitudes_1)],
                             'amplitudes_1_median_' + str(lead): [compute_median(amplitudes_1)],
                             'amplitudes_1_std_' + str(lead): [compute_std(amplitudes_1)],
                             })
    if lead == 6:
        QRSon = features_dict['QRSon']
        QRSoff = features_dict['QRSoff']
        indexes_effectives_QRS = QRSon * QRSoff > 0
        QRSon_duration, QRSoff_duration = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        amplitudes_1 = ecg[np.asarray(QRSoff_duration) + 30]
        feat = pd.DataFrame({'amplitudes_1_max_' + str(lead): [maximum(amplitudes_1)],
                             'amplitudes_1_mean_' + str(lead): [compute_mean(amplitudes_1)],
                             'amplitudes_1_median_' + str(lead): [compute_median(amplitudes_1)],
                             'amplitudes_1_std_' + str(lead): [compute_std(amplitudes_1)],
                             })
    if lead == 8:
        QRSoff = features_dict['QRSoff']
        Ton = features_dict['Ton']
        indexes_ST = QRSoff * Ton > 0
        longueur_ref = min(len(QRSoff), len(Ton))
        QRSoff_here = QRSoff[:longueur_ref]
        Ton_here = Ton[:longueur_ref]
        indexes_ST_here = indexes_ST[:longueur_ref]
        QRSoff_ST, Ton = QRSoff_here[indexes_ST_here], Ton_here[indexes_ST_here]
        ST_segments = [ecg[QRSoff_ST[i]:Ton[i]] for i in range(len(QRSoff_ST))]
        ST_segments_mean = compute_mean([compute_mean(ST_segments[i]) for i in range(len(ST_segments))])
        ST_segments_median = compute_median([compute_median(ST_segments[i]) for i in range(len(ST_segments))])
        ST_segments_std = compute_std([compute_std(ST_segments[i]) for i in range(len(ST_segments))])
        feat = pd.DataFrame({'ST_segment_mean_' + str(lead): [ST_segments_mean],
                             'ST_segment_median_' + str(lead): [ST_segments_median],
                             'ST_segment_std_' + str(lead): [ST_segments_std]})
    if lead == 10:
        QRSon = features_dict['QRSon']
        QRSoff = features_dict['QRSoff']
        indexes_effectives_QRS = QRSon * QRSoff > 0
        QRSon_duration, QRSoff_duration = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        amplitudes_1 = ecg[np.asarray(QRSoff_duration) + 30]
        feat = pd.DataFrame({'amplitudes_1_max_' + str(lead): [maximum(amplitudes_1)],
                             'amplitudes_1_mean_' + str(lead): [compute_mean(amplitudes_1)],
                             'amplitudes_1_median_' + str(lead): [compute_median(amplitudes_1)],
                             'amplitudes_1_std_' + str(lead): [compute_std(amplitudes_1)],
                             })
    if lead == 11:
        QRSon = features_dict['QRSon']
        QRSoff = features_dict['QRSoff']
        indexes_effectives_QRS = QRSon * QRSoff > 0
        QRSon_duration, QRSoff_duration = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        amplitudes_1 = ecg[np.asarray(QRSoff_duration) + 30]
        feat = pd.DataFrame({'amplitudes_1_max_' + str(lead): [maximum(amplitudes_1)],
                             'amplitudes_1_mean_' + str(lead): [compute_mean(amplitudes_1)],
                             'amplitudes_1_median_' + str(lead): [compute_median(amplitudes_1)],
                             'amplitudes_1_std_' + str(lead): [compute_std(amplitudes_1)],
                             })
    return feat
