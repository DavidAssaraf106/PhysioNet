import numpy as np
import pandas as pd
from Preprocessing_features_Submission import compute_std, compute_median, compute_mean, maximum, minimum


def extraction_feature_QAb(ecg, freq, features_dict, lead, preprocess=False):
    R_points = features_dict['R']
    Q_points = features_dict['Q']
    QRSon_points = features_dict['QRSon']
    indexes_effectives = R_points * Q_points * QRSon_points > 0
    Qamp = ecg[Q_points[indexes_effectives]] - ecg[QRSon_points[indexes_effectives]]
    Q_loc, R_loc, QRSon_loc = np.array(Q_points[indexes_effectives]), np.array(R_points[indexes_effectives]), np.array(QRSon_points[
        indexes_effectives])
    end_Qwave = np.squeeze([np.argmin(np.abs(np.asarray(ecg)[Q_loc[i]:R_loc[i]] - np.asarray(ecg)[QRSon_loc[i]])) for i in range(len(Q_loc))])
    Qdur = end_Qwave - Q_loc
    feat = pd.DataFrame({
        'Qamp_min_' + str(lead): [minimum((Qamp))],
        'Qamp_max_' + str(lead): [maximum((Qamp))],
        'Qamp_mean_' + str(lead): [compute_mean((Qamp))],
        'Qamp_median_' + str(lead): [compute_median((Qamp))],
        'Qamp_std_' + str(lead): [compute_std((Qamp))],
        'Qdur_min_' + str(lead): [minimum((Qdur))],
        'Qdur_max_' + str(lead): [maximum((Qdur))],
        'Qdur_mean_' + str(lead): [compute_mean((Qdur))],
        'Qdur_median_' + str(lead): [compute_median((Qdur))],
        'Qdur_std_' + str(lead): [compute_std((Qdur))],
    })
    return feat







