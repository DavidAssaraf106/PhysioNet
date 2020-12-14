import numpy as np
import pandas as pd
from Preprocessing_features_Submission import compute_mean


def extraction_feature_AD(ecg, freq, features_dict, lead, preprocess=False):
    if lead in {0, 1, 5}:
        R_points = features_dict['R']
        Q_points = features_dict['Q']
        S_points = features_dict['S']
        indexes_effectives = R_points * Q_points * S_points > 0
        QRS_deflection = ecg[R_points[indexes_effectives]] - np.abs(ecg[Q_points[indexes_effectives]]) - np.abs(
            ecg[S_points[indexes_effectives]])
        feat = pd.DataFrame({
            'Net_QRS_deflection_' + str(lead): [compute_mean(QRS_deflection)],
            'Sign_Net_QRS_deflection_' + str(lead): [int(compute_mean(QRS_deflection) > 0)]
        })
    else:
        feat = pd.DataFrame()
    return feat

