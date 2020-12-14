import os
import sys
import pandas as pd
import numpy as np
import matlab.engine
import pywt
from scipy.io import loadmat
# from skimage.restoration import denoise_wavelet

# from QRSDetectors import QRS_detector_gqrs, QRS_detector_xqrs, QRS_detector_PT

# insert your input directory and the matlab paths

if os.name == 'nt':
    data_extracted = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Complete_model\\"
    input_directory = "C:\\Users\\David\\PhysioNet_Code\\Training_WFDB"
    matlab_path1 = "C:\\Users\\David\\ecg-kit-master\\common\\wavedet"
    matlab_path2 = "C:\\Users\\David\\ecg-kit-master\\common"

if os.name == 'posix':
    data_extracted = "/home/david/Utils/Global_model/"
    input_directory = "/home/david/Training_WFDB"
    matlab_path1 = "/home/david/ecg-kit-master/common/wavedet"
    matlab_path2 = "/home/david/ecg-kit-master/common"

pathologies = 'AF_AVB_PAC_PVC_ST'

lead_to_func_ending_dict = {0: "I", 1: "II", 2: "III", 3: "aVR", 4: "aVL", 5: "aVF", 6: "V1", 7: "V2",
                            8: "V3",
                            9: "V4", 10: "V5", 11: "V6"}

interesting_keys = ["P", "Poff", "QRSon", "Q", "R", "S", "QRSoff", "Ton"]


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data

#todo: change le preprocessing
def preprocess_filter_wavelet(data, list_lead):
    data_bis = data.copy()  # .tolist()?
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        filtered_ecg_measurements = bandpass_filter(data[num_lead], lowcut=0.5, highcut=200,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=150,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=150,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
         # filtered_ecg_measurements = denoise_wavelet(filtered_ecg_measurements / 10000, wavelet='db4', sigma=0.008,
                                          #           rescale_sigma=True) * 10000
        filtered_ecg_measurements = bandpass_filter(data[num_lead], lowcut=0.01, highcut=200,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=150,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=80,
                                                    signal_freq=500, filter_order=3)  # 120 était good
        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
        data_bis[num_lead] = filtered_ecg_measurements
    return data_bis


def extraction_feature_wavedet(filename, list_lead, ecg, freq, eng, qrs_peak=[]):
    wavedet_3D_dict = wavedet_3D_wrapper(filename=filename, list_lead=list_lead, qrs_peak=qrs_peak, eng=eng, ecg=ecg)
    signal_DataFrame = pd.DataFrame()
    for i, lead in enumerate(list_lead):
        wavedet_3D_dict_i = wavedet_3D_dict[i]
        extracted_features = [(key, np.asarray(wavedet_3D_dict_i[key], dtype=np.int32)) for key in interesting_keys]
        dict_features = {key: value for key, value in extracted_features}
        feat_lead = extraction_feature_AF_AVB_PAC_PVC_ST(ecg[lead], freq, dict_features, lead)
        signal_DataFrame = pd.concat([signal_DataFrame, feat_lead], axis=1)
    return signal_DataFrame


def extraction_feature_AF_AVB_PAC_PVC_ST(ecg, freq, features_dict, lead):
    T = 3
    R_points = features_dict['R']
    R_locations = R_points[R_points > 0] #todo: améliorer la qualité de ce feature pour quand on a des zéros.
    Q_locations = features_dict['Q']
    S_locations = features_dict['S']
    P_locations = features_dict['P']
    indexes_effectives = Q_locations * S_locations > 0
    Q_locations = Q_locations[indexes_effectives]
    S_locations = S_locations[indexes_effectives]
    Q_locations_modif = Q_locations + T
    S_locations_modif = S_locations + T
    Dqrs = Q_locations - S_locations
    amplitudes = np.asarray(ecg[R_locations])
    DF = amplitudes[1:] - amplitudes[:-1]
    area = np.asarray([np.sum(ecg[Q_locations[i]:S_locations[i]]) for i in range(len(Q_locations))])
    Xh = np.asarray(
        [np.asarray(ecg[Q_locations[i]:S_locations[i]]) ** 2 for i in range(len(Q_locations))])  # array of arrays
    XhT = np.asarray([np.asarray(ecg[Q_locations_modif[i]:S_locations_modif[i]]) ** 2 for i in range(len(Q_locations))])
    d = np.asarray([np.sqrt(Xh[i] + XhT[i]) for i in range(len(Xh))])
    d = [maximum(d[i] - minimum(d[i])) for i in range(len(d))]
    indexes_effectives_2 = R_points * P_locations > 0
    RR_index = (R_points[:-1] * R_points[1:] > 0)
    R_points_effective, P_locations_effective = R_points[indexes_effectives_2], P_locations[indexes_effectives_2]
    PR_interval = R_points_effective - P_locations_effective
    RR_interval = np.asarray(comp_diff(R_points))
    final_index = RR_index * indexes_effectives_2[1:] > 0
    PR_interval_final = R_points[1:][final_index] - P_locations[1:][final_index]
    QRSon = features_dict['QRSon']
    QRSoff = features_dict['QRSoff']
    indexes_effectives_QRS = QRSon * QRSoff > 0
    QRSon, QRSoff = QRSon[indexes_effectives_QRS], QRSoff[indexes_effectives_QRS]
    R = features_dict['R']
    indexes_RR = R[1:] * R[:-1] > 0
    rr = np.asarray(comp_diff(R))
    rr = rr[indexes_RR]
    QRSWidth = calculate_QRS_Width(QRSon, QRSoff)
    QRSArea = calculate_QRS_Area(ecg, QRSon, QRSoff)
    IR = np.asarray(rr) / compute_mean(rr)
    SD1, SD2 = comp_poincare(rr)
    QRSon = features_dict['QRSon']
    QRSoff = features_dict['QRSoff']
    Ton = features_dict['Ton']
    indexes_ST = QRSon * Ton > 0
    indexes_QRS = QRSon * QRSoff > 0
    QRSon = np.asarray(QRSon[indexes_QRS])
    QRSoff_on = np.asarray(QRSoff[indexes_QRS])
    longueur_ref = min(len(QRSoff), len(Ton))
    QRSoff_here = QRSoff[:longueur_ref]
    Ton_here = Ton[:longueur_ref]
    indexes_ST_here = indexes_ST[:longueur_ref]
    QRSoff_ST, Ton = QRSoff_here[indexes_ST_here], Ton_here[indexes_ST_here]
    amplitudes_0 = ecg[QRSon]
    amplitudes_1 = ecg[QRSoff_on + 30]
    amplitudes_2 = ecg[QRSoff_on + 40]
    elevation_1 = amplitudes_1-amplitudes_0
    elevation_2 = amplitudes_2-amplitudes_0
    ST_segments = [ecg[QRSoff_ST[i]:Ton[i]] for i in range(len(QRSoff_ST))]
    ST_segments_mean = compute_mean([compute_mean(ST_segments[i]) for i in range(len(ST_segments))])
    ST_segments_median = compute_median([compute_median(ST_segments[i]) for i in range(len(ST_segments))])
    ST_segments_std = compute_std([compute_std(ST_segments[i]) for i in range(len(ST_segments))])
    if len(PR_interval_final) == len(RR_interval[final_index]):
        RAPR = compute_mean(PR_interval_final / RR_interval[final_index])
    else:
        RAPR = compute_mean(PR_interval_final[1:] / RR_interval[final_index])
    if len(R_points) > 5:
        # PVC detection:

            feat = pd.DataFrame({
                'Elevation1_mean'+str(lead):[compute_mean(elevation_1)],
                'Elevation1_median' + str(lead): [compute_median(elevation_1)],
                'Elevation1_max' + str(lead): [maximum(elevation_1)],
                'Elevation1_std' + str(lead): [compute_std(elevation_1)],
                'Elevation2_mean' + str(lead): [compute_mean(elevation_2)],
                'Elevation2_mean' + str(lead): [compute_median(elevation_2)],
                'Elevation2_mean' + str(lead): [maximum(elevation_2)],
                'Elevation2_mean' + str(lead): [compute_std(elevation_2)],
                'Fmax_' + str(lead): [maximum(amplitudes)],
                'Fmean' + str(lead): [compute_mean(amplitudes)],
                'Fmed' + str(lead): [compute_median(amplitudes)],
                'Fstd' + str(lead): [compute_std(amplitudes)],
                'DFmax' + str(lead): [maximum(DF)],
                'DFmean' + str(lead): [compute_mean(DF)],
                'DFmed' + str(lead): [compute_median(DF)],
                'DFstd' + str(lead): [compute_std(DF)],
                'Dqrsmax' + str(lead): [maximum(Dqrs)],
                'Dqrsmean' + str(lead): [compute_mean(Dqrs)],
                'Dqrsmed' + str(lead): [compute_median(Dqrs)],
                'Dqrsstd' + str(lead): [compute_std(Dqrs)],
                'Sqrsmax' + str(lead): [maximum(area)],
                'Sqrsmed' + str(lead): [compute_median(area)],
                'Sqrsmean' + str(lead): [compute_mean(area)],
                'Sqrssted' + str(lead): [compute_std(area)],
                'Amax' + str(lead): [maximum(d)],
                'Amedian' + str(lead): [compute_median(d)],
                'Amean' + str(lead): [compute_mean(d)],
                'Astd' + str(lead): [compute_std(d)],
                'IRmax'+ str(lead): [maximum(IR)],
                'IRmedian'+ str(lead): [compute_median(IR)],
                'IRmean'+ str(lead): [compute_mean(IR)],
                'IRstd'+ str(lead): [compute_std(IR)],

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
                'MDPR': [compute_median(PR_interval)],
                'MAPR': [compute_mean(PR_interval)],
                'RAPR': [RAPR],
                'RR_ratio_max' + str(lead): [maximum(rr)],
                'RR_ratio_median' + str(lead): [compute_median(rr)],
                'RR_ratio_mean' + str(lead): [compute_mean(rr)],
                'RR_ratio_std' + str(lead): [compute_std(rr)],
                'QRSWidth_max' + str(lead): [maximum(QRSWidth)],
                'QRSWidth_median' + str(lead): [compute_median(QRSWidth)],
                'QRSWidth_mean' + str(lead): [compute_mean(QRSWidth)],
                'QRSWidth_std' + str(lead): [compute_std(QRSWidth)],
                'QRSArea_max' + str(lead): [maximum(QRSArea)],
                'QRSArea_median' + str(lead): [compute_median(QRSArea)],
                'QRSArea_mean' + str(lead): [compute_mean(QRSArea)],
                'QRSArea_std' + str(lead): [compute_std(QRSArea)],
                'amplitudes_0_max' + str(lead): [maximum(amplitudes_0)],
                'amplitudes_0_mean' + str(lead): [compute_mean(amplitudes_0)],
                'amplitudes_0_median' + str(lead): [compute_median(amplitudes_0)],
                'amplitudes_0_std' + str(lead): [compute_std(amplitudes_0)],
                'amplitudes_1_max' + str(lead): [maximum(amplitudes_1)],
                'amplitudes_1_mean' + str(lead): [compute_mean(amplitudes_1)],
                'amplitudes_1_median' + str(lead): [compute_median(amplitudes_1)],
                'amplitudes_1_std' + str(lead): [compute_std(amplitudes_1)],
                'amplitudes_2_max' + str(lead): [maximum(amplitudes_2)],
                'amplitudes_2_mean' + str(lead): [compute_mean(amplitudes_2)],
                'amplitudes_2_median' + str(lead): [compute_median(amplitudes_2)],
                'amplitudes_2_std' + str(lead): [compute_std(amplitudes_2)],
                'ST_segment_mean' + str(lead): [ST_segments_mean],
                'ST_segment_median' + str(lead): [ST_segments_median],
                'ST_segment_std' + str(lead): [ST_segments_std]
            })

    else:

            feat = pd.DataFrame({
                'Elevation1_mean' + str(lead): [0],
                'Elevation1_median' + str(lead): [0],
                'Elevation1_max' + str(lead): [0],
                'Elevation1_std' + str(lead): [0],
                'Elevation2_mean' + str(lead): [0],
                'Elevation2_mean' + str(lead): [0],
                'Elevation2_mean' + str(lead): [0],
                'Elevation2_mean' + str(lead): [0],
                'Fmax_' + str(lead): [0],
                'Fmean' + str(lead): [0],
                'Fmed' + str(lead): [0],
                'Fstd' + str(lead): [0],
                'DFmax' + str(lead): [0],
                'DFmean' + str(lead): [0],
                'DFmed' + str(lead): [0],
                'DFstd' + str(lead): [0],
                'Dqrsmax' + str(lead): [0],
                'Dqrsmean' + str(lead): [0],
                'Dqrsmed' + str(lead): [0],
                'Dqrsstd' + str(lead): [0],
                'Sqrsmax' + str(lead): [0],
                'Sqrsmed' + str(lead): [0],
                'Sqrsmean' + str(lead): [0],
                'Sqrssted' + str(lead): [0],  # todo: type of area?
                'Amax' + str(lead): [0],
                'Amedian' + str(lead): [0],
                'Amean' + str(lead): [0],
                'Astd' + str(lead): [0],
                'IRmax': [0],
                'IRmedian': [0],
                'IRmean': [0],
                'IRstd': [0],
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
                'MDPR': [0],
                'MAPR': [0],
                'RAPR': [0],
                'RR_ratio_max' + str(lead): [0],
                'RR_ratio_median' + str(lead): [0],
                'RR_ratio_mean' + str(lead): [0],
                'RR_ratio_std' + str(lead): [0],
                'QRSWidth_max' + str(lead): [0],
                'QRSWidth_median' + str(lead): [0],
                'QRSWidth_mean' + str(lead): [0],
                'QRSWidth_std' + str(lead): [0],
                'QRSArea_max' + str(lead): [0],
                'QRSArea_median' + str(lead): [0],
                'QRSArea_mean' + str(lead): [0],
                'QRSArea_std' + str(lead): [0],
                'amplitudes_0_max' + str(lead): [0],
                'amplitudes_0_mean' + str(lead): [0],
                'amplitudes_0_median' + str(lead): [0],
                'amplitudes_0_std' + str(lead): [0],
                'amplitudes_1_max' + str(lead): [0],
                'amplitudes_1_mean' + str(lead): [0],
                'amplitudes_1_median' + str(lead): [0],
                'amplitudes_1_std' + str(lead): [0],
                'amplitudes_2_max' + str(lead): [0],
                'amplitudes_2_mean' + str(lead): [0],
                'amplitudes_2_median' + str(lead): [0],
                'amplitudes_2_std' + str(lead): [0],
                'ST_segment_mean' + str(lead): [0],
                'ST_segment_median' + str(lead): [0],
                'ST_segment_std' + str(lead): [0]
            })

    return feat


def extraction_feature_PVC_AVB_AF(ecg, freq, features_dict, lead):
    T = 3
    R_points = features_dict['R']
    R_locations = R_points[R_points > 0] #todo: améliorer quand on avait des zéros?
    Q_locations = features_dict['Q']
    S_locations = features_dict['S']
    P_locations = features_dict['P']
    indexes_effectives = Q_locations * S_locations > 0
    Q_locations = Q_locations[indexes_effectives]
    S_locations = S_locations[indexes_effectives]
    Q_locations_modif = Q_locations + T
    S_locations_modif = S_locations + T
    Dqrs = Q_locations - S_locations
    amplitudes = np.asarray(ecg[R_locations])
    DF = amplitudes[1:] - amplitudes[:-1]
    area = np.asarray([np.sum(ecg[Q_locations[i]:S_locations[i]]) for i in range(len(Q_locations))])
    Xh = np.asarray(
        [np.asarray(ecg[Q_locations[i]:S_locations[i]]) ** 2 for i in range(len(Q_locations))])  # array of arrays
    XhT = np.asarray([np.asarray(ecg[Q_locations_modif[i]:S_locations_modif[i]]) ** 2 for i in range(len(Q_locations))])
    d = np.asarray([np.sqrt(Xh[i] + XhT[i]) for i in range(len(Xh))])
    d = [maximum(d[i] - minimum(d[i])) for i in range(len(d))]
    rr = comp_diff(R_locations)
    IR = np.asarray(rr)/compute_mean(rr)
    rr = comp_diff(R_points)
    SD1, SD2 = comp_poincare(rr)
    indexes_effectives_2 = R_points * P_locations > 0
    RR_index = (R_points[:-1] * R_points[1:] > 0)

    R_points_effective, P_locations_effective = R_points[indexes_effectives_2], P_locations[indexes_effectives_2]
    PR_interval = R_points_effective - P_locations_effective
    RR_interval = np.asarray(comp_diff(R_points))
    final_index = RR_index * indexes_effectives_2[1:] > 0
    PR_interval_final = R_points[1:][final_index] - P_locations[1:][final_index]
    if len(PR_interval_final) == len(RR_interval[final_index]):
        RAPR = compute_mean(PR_interval_final / RR_interval[final_index])
    else:
        RAPR = compute_mean(PR_interval_final[1:] / RR_interval[final_index])
    if len(R_points) > 5:
        # PVC detection:
        if lead ==1:
            feat = pd.DataFrame({
                'Fmax_' + str(lead): [maximum(amplitudes)],
                'Fmean' + str(lead): [compute_mean(amplitudes)],
                'Fmed' + str(lead): [compute_median(amplitudes)],
                'Fstd' + str(lead): [compute_std(amplitudes)],
                'DFmax' + str(lead): [maximum(DF)],
                'DFmean' + str(lead): [compute_mean(DF)],
                'DFmed' + str(lead): [compute_median(DF)],
                'DFstd' + str(lead): [compute_std(DF)],
                'Dqrsmax' + str(lead): [maximum(Dqrs)],
                'Dqrsmean' + str(lead): [compute_mean(Dqrs)],
                'Dqrsmed' + str(lead): [compute_median(Dqrs)],
                'Dqrsstd' + str(lead): [compute_std(Dqrs)],
                'Sqrsmax' + str(lead): [maximum(area)],
                'Sqrsmed' + str(lead): [compute_median(area)],
                'Sqrsmean' + str(lead): [compute_mean(area)],
                'Sqrssted' + str(lead): [compute_std(area)],
                'Amax' + str(lead): [maximum(d)],
                'Amedian' + str(lead): [compute_median(d)],
                'Amean' + str(lead): [compute_mean(d)],
                'Astd' + str(lead): [compute_std(d)],
                'IRmax'+ str(lead): [maximum(IR)],
                'IRmedian'+ str(lead): [compute_median(IR)],
                'IRmean'+ str(lead): [compute_mean(IR)],
                'IRstd'+ str(lead): [compute_std(IR)],

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
                'MDPR': [compute_median(PR_interval)],
                'MAPR': [compute_mean(PR_interval)],
                'RAPR': [RAPR]
            })
        else:
            feat = pd.DataFrame({
                'Fmax_' + str(lead): [maximum(amplitudes)],
                'Fmean' + str(lead): [compute_mean(amplitudes)],
                'Fmed' + str(lead): [compute_median(amplitudes)],
                'Fstd' + str(lead): [compute_std(amplitudes)],
                'DFmax' + str(lead): [maximum(DF)],
                'DFmean' + str(lead): [compute_mean(DF)],
                'DFmed' + str(lead): [compute_median(DF)],
                'DFstd' + str(lead): [compute_std(DF)],
                'Dqrsmax' + str(lead): [maximum(Dqrs)],
                'Dqrsmean' + str(lead): [compute_mean(Dqrs)],
                'Dqrsmed' + str(lead): [compute_median(Dqrs)],
                'Dqrsstd' + str(lead): [compute_std(Dqrs)],
                'Sqrsmax' + str(lead): [maximum(area)],
                'Sqrsmed' + str(lead): [compute_median(area)],
                'Sqrsmean' + str(lead): [compute_mean(area)],
                'Sqrssted' + str(lead): [compute_std(area)],
                'Amax' + str(lead): [maximum(d)],
                'Amedian' + str(lead): [compute_median(d)],
                'Amean' + str(lead): [compute_mean(d)],
                'Astd' + str(lead): [compute_std(d)],
                'IRmax' + str(lead): [maximum(IR)],
                'IRmedian' + str(lead): [compute_median(IR)],
                'IRmean' + str(lead): [compute_mean(IR)],
                'IRstd' + str(lead): [compute_std(IR)]


            })
    else:
        if lead ==1:
            feat = pd.DataFrame({
                'Fmax_' + str(lead): [0],
                'Fmean' + str(lead): [0],
                'Fmed' + str(lead): [0],
                'Fstd' + str(lead): [0],
                'DFmax' + str(lead): [0],
                'DFmean' + str(lead): [0],
                'DFmed' + str(lead): [0],
                'DFstd' + str(lead): [0],
                'Dqrsmax' + str(lead): [0],
                'Dqrsmean' + str(lead): [0],
                'Dqrsmed' + str(lead): [0],
                'Dqrsstd' + str(lead): [0],
                'Sqrsmax' + str(lead): [0],
                'Sqrsmed' + str(lead): [0],
                'Sqrsmean' + str(lead): [0],
                'Sqrssted' + str(lead): [0],  # todo: type of area?
                'Amax' + str(lead): [0],
                'Amedian' + str(lead): [0],
                'Amean' + str(lead): [0],
                'Astd' + str(lead): [0],
                'IRmax': [0],
                'IRmedian': [0],
                'IRmean': [0],
                'IRstd': [0],
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
                'MDPR': [0],
                'MAPR': [0],
                'RAPR': [0]
            })
        else:
            feat = pd.DataFrame({
                'Fmax_' + str(lead): [0],
                'Fmean' + str(lead): [0],
                'Fmed' + str(lead): [0],
                'Fstd' + str(lead): [0],
                'DFmax' + str(lead): [0],
                'DFmean' + str(lead): [0],
                'DFmed' + str(lead): [0],
                'DFstd' + str(lead): [0],
                'Dqrsmax' + str(lead): [0],
                'Dqrsmean' + str(lead): [0],
                'Dqrsmed' + str(lead): [0],
                'Dqrsstd' + str(lead): [0],
                'Sqrsmax' + str(lead): [0],
                'Sqrsmed' + str(lead): [0],
                'Sqrsmean' + str(lead): [0],
                'Sqrssted' + str(lead): [0],  # todo: type of area?
                'Amax' + str(lead): [0],
                'Amedian' + str(lead): [0],
                'Amean' + str(lead): [0],
                'Astd' + str(lead): [0],
                'IRmax': [0],
                'IRmedian': [0],
                'IRmean': [0],
                'IRstd': [0]
            })
    return feat


def wavedet_3D_wrapper(filename, list_lead, qrs_peak, eng, ecg):  # engine does not support structure array
    ecg_matlab = []
    for i in range(len(ecg)):
        ecg_matlab.append(ecg[i].tolist())
    ret_val = eng.python_wrap_wavedet_3D(input_directory, matlab.double(ecg_matlab), filename, list_lead, [],
                                         nargout=12)
    for i, lead in enumerate(list_lead):
        if qrs_peak == []:
            qrs_peak_i = np.array(ret_val[i]["R"])
            qrs_peak_i[np.isnan(qrs_peak_i)] = 0
            qrs_peak_i = np.asarray(qrs_peak_i, dtype=np.int64)
        if len(qrs_peak_i[qrs_peak_i > 0]) > 5:
            for key in (ret_val[i].keys()):
                if interesting_keys.__contains__(key):
                    ret_val[i][key] = np.array(ret_val[i][key]._data).reshape(ret_val[i][key].size, order='F')[0]
                    ret_val[i][key][np.isnan(ret_val[i][key])] = 0
                    ret_val[i][key] = np.asarray(ret_val[i][key][1:], dtype=np.int64)
            R_ref = ret_val[i]['R']
            Poff = ret_val[i]['Poff']
            Ton = ret_val[i]['Ton']
            if interesting_keys.__contains__('Q'):
                for p, ind in enumerate(ret_val[i]['QRSon']):
                    ecg_lead = ecg[lead]
                    if R_ref[p] > 0:
                        if ind > 0 and ret_val[i]['Q'][p] == 0:
                            if R_ref[p] > ind + 1:
                                candidate = np.argmin(ecg_lead[ind:R_ref[p]]) + ind  # todo: à remplacer par compute_argmin?
                                if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                    ref_value = np.max(np.abs(ecg_lead[ind:ind + 5]))

                                    if Poff[p] > 0:
                                        indice_minimal = (candidate + Poff[p]) / 2
                                    else:
                                        indice_minimal = 0
                                    while ind > 0 and np.abs((ecg_lead[
                                                                  ind] - ref_value) / ref_value) < 0.2 and ind > indice_minimal:  # todo: divide by zero encountered
                                        ind = ind - 1
                                    ret_val[i]['QRSon'][p] = ind
                                ret_val[i]['Q'][p] = candidate  # in order to correct when we do not detect any Q points
                            else:
                                continue
            if interesting_keys.__contains__('S') :
                for p, ind in enumerate(ret_val[i]['QRSoff']):
                    ecg_lead = ecg[lead]
                    if R_ref[p] > 0:
                        if ind > 0 and ret_val[i]['S'][p] == 0:
                            if ind > R_ref[p] + 1:
                                candidate = np.argmin(ecg_lead[R_ref[p]:ind]) + R_ref[
                                    p]  # todo: à remplacer par compute_argmin?
                                if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                    ref_value = np.max(np.abs(ecg_lead[ind - 5:ind]))

                                    if Ton[p] > 0:
                                        indice_maximal = (candidate + Ton[p]) / 2
                                    else:
                                        indice_maximal = len(ecg_lead)
                                    while ind < len(ecg_lead) and (
                                            np.abs((ecg_lead[ind] - ref_value) / ref_value)) and ind < indice_maximal:
                                        ind = ind + 1
                                    ret_val[i]['QRSoff'][p] = ind
                                ret_val[i]['S'][p] = candidate  # in order to correct when we do not detect any S points
                            else:
                                continue
        else:
            for key in (ret_val[i].keys()):
                ret_val[i][key] = np.asarray([0])
            print("We are having troubles with", filename + ' lead ', lead)
    return ret_val


def get_features_from_QRS():
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_path1, nargout=0)
    eng.addpath(matlab_path2, nargout=0)
    input_files = []
    i = 0
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
            i += 1
    print(i, "signals to process")
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11]  # every lead, and then we will find the one maximizing differences
    for i, f in enumerate(input_files):
        if i < 0:
            continue
        else:
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            freq = 500
            ecg = data
            for iline in header_data:
                if iline.startswith('#Age'):
                    tmp_age = iline.split(': ')[1].strip()
                    age = int(tmp_age if tmp_age != 'NaN' else 57)
                elif iline.startswith('#Sex'):
                    tmp_sex = iline.split(': ')[1]
                    if tmp_sex.strip() == 'Female':
                        sex = 1
                    else:
                        sex = 0
                elif iline.startswith('#Dx'):
                    label = iline.split(': ')[1].split(',')[0].strip()  # single label classification
            DataFrame_sample_i = extraction_feature_wavedet(filename=input_files[i], list_lead=list_lead, ecg=ecg,
                                                            freq=freq, eng=eng)
            DataFrame_sample_i['Age'] = age
            DataFrame_sample_i['Sex'] = sex
            DataFrame_sample_i['Label'] = label
            if i == 0:
                DataFrame_sample_i.to_csv(data_extracted + "features_"+pathologies+".csv", index=False, header=True)
            else:
                DataFrame_sample_i.to_csv(data_extracted + "features_"+pathologies+".csv", index=False, header=False,
                                          mode='a')
            print(i + 1, "signals processed")
    eng.quit()


def boxplot():  # todo: change names of the features
    data_file = data_extracted + "features_"+pathologies+".csv"
    data_PVC = pd.read_csv(data_file)
    for i in range(len(data_PVC.columns)-3):
        data_PVC.boxplot(column=data_PVC.columns[i], by='Label', grid=False)
        plt.savefig(data_extracted + pathologies +'_boxplot_'+data_PVC.columns[i]+'.png')
        plt.cla()
        plt.clf()
        plt.close()




if __name__ == '__main__':
    get_features_from_QRS()

