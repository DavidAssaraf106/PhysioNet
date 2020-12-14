import numpy as np
import pandas as pd
from Preprocessing_features_Submission import compute_std, compute_median, compute_mean, maximum, minimum, bandpass_filter


def calculate_QRS_Width(QRSOn, QRSOff):
    """
    Automatic detection of premature atrial contractions in the electrocardiogram Vessela T. Krasteva, Irena I. Jekova, Ivaylo I. Christov
    """
    QRS_Width = np.asarray(np.asarray(QRSOff) - np.asarray(QRSOn))
    Median_5_Width = []
    for i in range(5, len(QRSOn)):
        Median_5_Width.append(compute_median(QRS_Width[i - 5:i]))
    Median_5_Width = np.asarray(Median_5_Width)
    if len(QRS_Width) < 6:
        return [0]
    return (np.abs(QRS_Width[5:] - Median_5_Width) / Median_5_Width) * 100


def calculate_QRS_Area(ecg, QRSOn, QRSOff):
    """
    Automatic detection of premature atrial contractions in the electrocardiogram Vessela T. Krasteva, Irena I. Jekova, Ivaylo I. Christov
    """
    QRS_Area = np.zeros(len(QRSOn))
    for i in range(len(QRSOn)):
        QRS_Area[i] = np.sum(np.abs(ecg[QRSOn[i]: QRSOff[i]]))
    Median_5_Area = []
    for i in range(5, len(QRSOn)):
        Median_5_Area.append(compute_median(QRS_Area[i - 5:i]))
    Median_5_Area = np.asarray(Median_5_Area)
    if len(QRS_Area) < 6:
        return [0]
    return (np.abs(QRS_Area[5:] - Median_5_Area) / Median_5_Area) * 100


def comp_diff(R_points):  # @Jeremy, for PVC detection
    R_points = np.asarray(R_points)
    cnt_diff_ecg = []
    for idx_q in range(1, len(R_points)):
        cnt_diff = R_points[idx_q] - R_points[idx_q - 1]
        cnt_diff_ecg.append(cnt_diff)
    return cnt_diff_ecg


def extraction_feature_PAC(ecg, freq, features_dict, lead, preprocess=False):
    QRSon = features_dict['QRSon']
    QRSoff = features_dict['QRSoff']
    indexes_effectives_QRS = QRSon * QRSoff > 0
    feat = pd.DataFrame()
    if lead == 1:
        R = features_dict['R']
        indexes_RR = R[1:] * R[:-1] > 0
        rr = np.asarray(comp_diff(R))
        rr = rr[indexes_RR]
        feat = pd.DataFrame({'RR_ratio_max_' + str(lead): [maximum(rr)]})
    if lead == 2:
        QRSon, QRSoff = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        QRSWidth = calculate_QRS_Width(QRSon, QRSoff)
        feat = pd.DataFrame({'QRSWidth_median_' + str(lead): [compute_median(QRSWidth)]})
    if lead == 4:
        R = features_dict['R']
        indexes_RR = R[1:] * R[:-1] > 0
        rr = np.asarray(comp_diff(R))
        rr = rr[indexes_RR]
        feat = pd.DataFrame({'RR_ratio_max_' + str(lead): [maximum(rr)],
                             'RR_ratio_median_' + str(lead): [compute_median(rr)],
                             'RR_ratio_mean_' + str(lead): [compute_mean(rr)]})
    if lead == 7:
        QRSon, QRSoff = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        QRSArea = calculate_QRS_Area(ecg, QRSon, QRSoff)
        feat = pd.DataFrame({'QRSArea_max_' + str(lead): [maximum(QRSArea)],
                             'QRSArea_median_' + str(lead): [compute_median(QRSArea)],
                             'QRSArea_mean_' + str(lead): [compute_mean(QRSArea)],
                             'QRSArea_std_' + str(lead): [compute_std(QRSArea)]
                             })
    if lead == 8:
        R = features_dict['R']
        indexes_RR = R[1:] * R[:-1] > 0
        rr = np.asarray(comp_diff(R))
        rr = rr[indexes_RR]
        feat = pd.DataFrame({'RR_ratio_median_' + str(lead): [compute_median(rr)],
                             'RR_ratio_mean_' + str(lead): [compute_mean(rr)]})
    if lead == 10:
        R = features_dict['R']
        indexes_RR = R[1:] * R[:-1] > 0
        rr = np.asarray(comp_diff(R))
        rr = rr[indexes_RR]
        feat = pd.DataFrame({'RR_ratio_max_' + str(lead): [maximum(rr)],
                             'RR_ratio_std_' + str(lead): [compute_std(rr)]})
    if lead == 11:
        QRSon, QRSoff = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
        QRSArea = calculate_QRS_Area(ecg, QRSon, QRSoff)
        QRSWidth = calculate_QRS_Width(QRSon, QRSoff)
        feat = pd.DataFrame({'QRSWidth_median_' + str(lead): [compute_median(QRSWidth)],
                             'QRSWidth_mean_' + str(lead): [compute_mean(QRSWidth)],
                             'QRSWidth_std_' + str(lead): [compute_std(QRSWidth)],
                             'QRSArea_max_' + str(lead): [maximum(QRSArea)],
                             'QRSArea_std_' + str(lead): [compute_std(QRSArea)]})
    return feat

