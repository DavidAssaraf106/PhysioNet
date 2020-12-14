import numpy as np
from Preprocessing_features_Submission import compute_median, compute_mean
import pandas as pd



def comp_diff(R_points):  # @Jeremy, for PVC detection
    R_points = np.asarray(R_points)
    cnt_diff_ecg = []
    for idx_q in range(1, len(R_points)):
        cnt_diff = R_points[idx_q] - R_points[idx_q - 1]
        cnt_diff_ecg.append(cnt_diff)
    return cnt_diff_ecg



def extraction_feature_AVB(ecg, freq, features_dict, lead, preprocess=False):
    # I-AVB detection:
    feat = pd.DataFrame()
    if lead == 11:
        R_points = features_dict['R']
        P_locations = features_dict['P']
        indexes_effectives = R_points * P_locations > 0
        RR_index = (R_points[:-1] * R_points[1:] > 0)
        R_points_effective, P_locations_effective = R_points[indexes_effectives], P_locations[indexes_effectives]
        PR_interval = R_points_effective - P_locations_effective
        RR_interval = np.asarray(comp_diff(R_points))
        final_index = RR_index * indexes_effectives[1:] > 0
        PR_interval_final = R_points[1:][final_index] - P_locations[1:][final_index]
        if len(PR_interval_final) == len(RR_interval[final_index]):
            RAPR = compute_mean(PR_interval_final / RR_interval[final_index])
        else:
            RAPR = compute_mean(PR_interval_final[1:] / RR_interval[final_index])
        feat = pd.DataFrame({'MDPR_'+str(lead): [compute_median(PR_interval)],
                             'MAPR_' + str(lead): [compute_mean(PR_interval)],
                             'RAPR_'+str(lead): [RAPR]})
    if lead == 5:
        R_points = features_dict['R']
        P_locations = features_dict['P']
        indexes_effectives = R_points * P_locations > 0
        RR_index = (R_points[:-1] * R_points[1:] > 0)
        R_points_effective, P_locations_effective = R_points[indexes_effectives], P_locations[indexes_effectives]
        PR_interval = R_points_effective - P_locations_effective
        RR_interval = np.asarray(comp_diff(R_points))
        final_index = RR_index * indexes_effectives[1:] > 0
        PR_interval_final = R_points[1:][final_index] - P_locations[1:][final_index]
        if len(PR_interval_final) == len(RR_interval[final_index]):
            RAPR = compute_mean(PR_interval_final / RR_interval[final_index])
        else:
            RAPR = compute_mean(PR_interval_final[1:] / RR_interval[final_index])
        feat = pd.DataFrame({'MDPR_'+str(lead): [compute_median(PR_interval)],
                             'MAPR_'+str(lead): [compute_mean(PR_interval)],
                             'RAPR_'+str(lead): [RAPR]})
    return feat







