import random
import warnings
import csv
import numpy as np
import wfdb.processing
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import physionet_matlab
import scipy
from scipy.signal import savgol_filter, butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial import cKDTree
from Preprocessing_features import compute_mean
from ST import extraction_feature_ST
from AF import extraction_feature_AF
from AVB import extraction_feature_AVB
from PAC import extraction_feature_PAC
from PVC import extraction_feature_PVC
from Bundle_Branch_Block import extraction_feature_wavedet_BBB
from Scoring_file import compute_challenge_metric, load_weights, compute_confusion_matrices, compute_beta_measures, \
    compute_modified_confusion_matrix

warnings.filterwarnings("ignore")

if os.name == 'nt':
    input_directory = "C:\\Users\\David\\PhysioNet_Code\\Training_complete"
    matlab_path1 = "C:\\Users\\David\\ecg-kit-master\\common\\wavedet"
    matlab_path2 = "C:\\Users\\David\\ecg-kit-master\\common"
    features_location = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Experiments\\New_Features\\Experiment_new_features\\Incremental_results\\training_database.csv"
    training_location = 'C:\\Users\\David\PhysioNet_Code\\First_Classifier\\Utils\\Experiments\\New_Features\\Experiment_new_features\\Incremental_results\\train_filenames.csv'
    testing_location = 'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Incremental_results\\Features\\test_filenames.csv'
    test = 'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Experiments\\New_Features\\Experiment_new_features\\Incremental_results\\test_compilation.csv'

if os.name == 'posix':
    test_set_location = '/home/david/features_test_multilabelled.csv'
    input_directory = "/home/david/Training_WFDB_new"
    matlab_path1 = "/home/david/ecg-kit-master/common/wavedet"
    testing_location = '/home/david/test_filenames.csv'
    matlab_path2 = "/home/david/ecg-kit-master/common"

lead_to_func_ending_dict = {0: "I", 1: "II", 2: "III", 3: "aVR", 4: "aVL", 5: "aVF", 6: "V1", 7: "V2",
                            8: "V3", 9: "V4", 10: "V5", 11: "V6"}
interesting_keys = ["Pon", "P", "Poff", "QRSon", "Q", "R", "S", "QRSoff", "Ton", "T", "Toff", "Ttipo", "Ttipoon",
                    "Ttipooff"]
annot_to_patho = {'270492004': 'IAVB',
                  '195042002': 'IIAVB',
                  '164951009': 'abQRS',
                  '61277005': 'AIVR',
                  '426664006': 'AJR',
                  '413444003': 'AMIs',
                  '426434006': 'AnMIs',
                  '54329005': 'AnMI',
                  '164889003': 'AF',
                  '195080001': 'AFAFL',
                  '164890007': 'AFL',
                  '195126007': 'AH',
                  '251268003': 'AP',
                  '713422000': 'ATach',
                  '233917008': 'AVB',
                  '251170000': 'BPAC',
                  '74615001': 'BTS',
                  '426627000': 'Brady',
                  '418818005': 'Brug',
                  '6374002': 'BBB',
                  '426749004': 'CAF',
                  '413844008': 'CMIs',
                  '78069008': 'CRPC',
                  '27885002': 'CHB',
                  '713427006': 'RBBB',
                  '77867006': 'SQT',
                  '82226007': 'DIVB',
                  '428417006': 'ERe',
                  '251143007': 'ART',
                  '29320008': 'ER',
                  '423863005': 'EA',
                  '251259000': 'HTV',
                  '251120003': 'ILBBB',
                  '713426002': 'IRBBB',
                  '251200008': 'ICA',
                  '425419005': 'IIs',
                  '704997005': 'ISTD',
                  '50799005': 'IAVD',
                  '426995002': 'JE',
                  '251164006': 'JPC',
                  '426648003': 'JTach',
                  '425623009': 'LIs',
                  '445118002': 'LAnFB',
                  '253352002': 'LAA',
                  '67741000119109': 'LAE',
                  '446813000': 'LAH',
                  '39732003': 'LAD',
                  '164909002': 'LBBB',
                  '445211001': 'LPFB',
                  '164873001': 'LVH',
                  '370365005': 'LVS',
                  '251146004': 'LQRSV',
                  '251147008': 'LQRSVLL',
                  '251148003': 'LQRSP',
                  '28189009': 'MoII',
                  '54016002': 'MoI',
                  '713423005': 'MATach',
                  '164865005': 'MI',
                  '164861001': 'MIs',
                  '65778007': 'NSIACB',
                  '698252002': 'NSIVCB',
                  '428750005': 'NSSTTA',
                  '164867002': 'OldMI',
                  '10370003': 'PR',
                  '67198005': 'PSVT',
                  '164903001': 'PAVB21',
                  '284470004': 'PAC',
                  '164884008': 'NotPVC',
                  '427172004': 'PVC',
                  '111975006': 'LQT',
                  '164947007': 'LPR',
                  '164917005': 'QAb',
                  '164921003': 'RAb',
                  '253339007': 'RAAb',
                  '446358003': 'RAH',
                  '47665007': 'RAD',
                  '59118001': 'RBBB',
                  '89792004': 'RVH',
                  '55930002': 'STC',
                  '49578007': 'SPRI',
                  '427393009': 'SA',
                  '426177001': 'SB',
                  '426783006': 'SNR',
                  '427084000': 'STach',
                  '429622005': 'STD',
                  '164931005': 'STE',
                  '164930006': 'STIAb',
                  '63593006': 'PAC',
                  '426761007': 'SVT',
                  '251139008': 'ALR',
                  '164934002': 'TAb',
                  '59931005': 'TInv',
                  '251242005': 'UTall',
                  '164937009': 'UAb',
                  '11157007': 'VBig',
                  '17338001': 'PVC',
                  '75532003': 'VEsB',
                  '81898007': 'VEsR',
                  '164896001': 'VF',
                  '111288001': 'VFL',
                  '266249003': 'VH',
                  '251266004': 'VPP',
                  '195060002': 'VPEx',
                  '164895002': 'VTach',
                  '251180001': 'VTrig',
                  '195101003': 'WAP',
                  '74390002': 'WPW'}  # contains updates on SNOMED-CT codes
test_pathologies_effective = ['164889003', '164890007', '426627000', '270492004', '713426002', '445118002',
                              '39732003',
                              '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007',
                              '111975006',
                              '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000',
                              '164934002', '59931005']
pathologies_all = ['AF', 'AFL', 'Brady', 'RBBB', 'IAVB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR',
                   'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'SNR', 'STach', 'PAC', 'TAb', 'TInv',
                   'PVC']
pathologies_effective = ['AF', 'AFL', 'Brady', 'IAVB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR',
                         'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'SNR', 'STach', 'TAb', 'TInv']
hr_pathologies = ['CRBBB', 'LPR', 'PR', 'SVPB']
q_pathologies = ['Brady', 'PVC']
test_pathologies_all = ['164889003', '164890007', '426627000', '713427006', '270492004', '713426002', '445118002',
                        '39732003',
                        '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007',
                        '111975006',
                        '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000',
                        '63593006',
                        '164934002', '59931005', '17338001']
scored_pathologies = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LPR', 'LQRSV',
                      'LQT', 'NSIVCB', 'PAC', 'PVC', ' PR', 'QAb', 'RAD',
                      'RBBB', 'SA', 'SB', 'SNR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB']
normal = '426783006'



##################################################
############# Data Processing & FE ###############
##################################################


def get_gain_lead(header_data):
    tmp_hea = header_data[0].split(' ')
    num_leads = int(tmp_hea[1])
    gain_lead = np.zeros(num_leads)
    for ii in range(num_leads):
        tmp_hea = header_data[ii + 1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])
    return gain_lead


def get_freq(header_data):
    for lines in header_data:
        tmp = lines.split(' ')
        return int(tmp[2])


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """This function uses a Butterworth filter. The coefficoents are computed automatically. Lowcut and highcut are in Hz"""
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def get_label(integer):
    return annot_to_patho[str(integer)]


def load_challenge_data(filename):
    """Loads the data from the files (require .hea and .mat files)"""
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def extraction_feature_wavedet(filename, list_lead, ecg, freq, runtime, qrs_peak=[]):
    """This function processes the output of wavedet into a dataframe, for all of the leads considered"""
    wavedet_3D_dict, wavedet_3D_dict_challenge = wavedet_3D_wrapper(filename=filename, list_lead=list_lead,
                                                                    qrs_peak=qrs_peak, runtime=runtime, ecg=ecg)
    challenge_results = challenge_wrapper(ecg=ecg, list_lead=list_lead, runtime=runtime, dict=wavedet_3D_dict_challenge)
    signal_DataFrame = pd.DataFrame()
    for i, lead in enumerate(list_lead):
        wavedet_3D_dict_i = wavedet_3D_dict[i]
        extracted_features = [(key, np.asarray(wavedet_3D_dict_i[key], dtype=np.int32)) for key in interesting_keys]
        dict_features = {key: value for key, value in extracted_features}
        feat_lead = extraction_feature_final(ecg[lead], freq, dict_features, lead)
        signal_DataFrame = pd.concat([signal_DataFrame, feat_lead], axis=1)
    signal_DataFrame = pd.concat([signal_DataFrame, challenge_results], axis=1)
    return signal_DataFrame


def challenge_wrapper(ecg, list_lead, runtime, dict):
    """This function processes the output of the challenge matlab function, into a dataframe. The challenge function required
    qrs points. Inside this function, we also compute the bsqi index, which allows later on to discard ecgs."""
    challenge_DataFrame = pd.DataFrame()
    for i, lead in enumerate(list_lead):
        points_i = dict[i]
        qrs_points = np.array(points_i["R"])
        qrs_points[np.isnan(qrs_points)] = 0
        qrs_points = np.asarray(qrs_points, dtype=np.int64)
        R_points = wfdb.processing.gqrs_detect(fs=500, sig=ecg[lead], adc_gain=1000, adc_zero=0)
        ind = bsqi(qrs_points, R_points)
        challenge_result_i = runtime.challenge(ecg[lead].tolist(), lead, 500, qrs_points.tolist(), dict[lead])
        challenge_result_i['bsqi_' + str(lead)] = ind
        for key, values in challenge_result_i.items():
            if np.isnan(values):
                values = 0
            challenge_result_i[key] = [values]
        challenge_result_i = pd.DataFrame.from_dict(challenge_result_i)
        challenge_DataFrame = pd.concat([challenge_DataFrame, challenge_result_i], axis=1)
    return challenge_DataFrame


def wavedet_3D_wrapper(filename, list_lead, qrs_peak, runtime, ecg):  # engine does not support structure array
    """Gets the output of the wavedet algorithm on Matlab. The raw wavedet algorithm had issues with detecting Q and S
    points so this is something I edited manually"""
    ecg_matlab = []
    for i in range(len(ecg)):
        ecg_matlab.append(ecg[i].tolist())
    ret_val = runtime.python_wrap_wavedet_3D(input_directory, ecg_matlab, filename, list_lead, [],
                                         nargout=12)
    ret_val_matlab = [dict(ret_val[i]) for i in range(len(ret_val))]
    print(ret_val)
    for i, lead in enumerate(list_lead):
        for key in (ret_val[i].keys()):
            if interesting_keys.__contains__(key):
                ret_val[i][key] = np.array(ret_val[i][key]._data).reshape(ret_val[i][key].size, order='F')[0]
                ret_val[i][key][np.isnan(ret_val[i][key])] = 0
                ret_val[i][key] = np.asarray(ret_val[i][key][1:], dtype=np.int64)
        R_ref = ret_val[i]['R']
        Poff = ret_val[i]['Poff']
        Ton = ret_val[i]['Ton']
        if interesting_keys.__contains__('Q'):   # choice
            for p, ind in enumerate(ret_val[i]['QRSon']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['Q'][p] == 0:
                        if R_ref[p] > ind + 1:
                            candidate = np.argmin(
                                ecg_lead[ind:R_ref[p]]) + ind  # todo: à remplacer par compute_argmin?
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
        if interesting_keys.__contains__('S'):  # choice
            for p, ind in enumerate(ret_val[i]['QRSoff']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['S'][p] == 0:
                        if ind > R_ref[p] + 1:
                            candidate = np.argmin(ecg_lead[R_ref[p]:ind]) + R_ref[
                                p]
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
    return ret_val, ret_val_matlab


def ecgtovcg(signal):
    """
    This function implementation is based on the paper: Linear affine transformations
    between 3-lead (Frank XYZ leads) vectorcardiogram and 12-lead electrocardiogram signals,
    Drew Dawson, Hui Yang, Milind Malshe.
    We convert an 8-leads ecg (leads I, II, V1, V2, V3, V4, V5, V6 to a planar VCG representation.
    We will only use a single plane projection of this VCG representation.
    Input: 12-leads ecg.
    Output: Planar representation of the ecg.
    """
    leads_to_be_transformed = [0, 1, 6, 7, 8, 9, 10, 11]
    ecg_toconvert = np.asarray(np.asarray(signal)[leads_to_be_transformed])
    # I       II      V1      V2     V3     V4     V5     V6
    invDower = np.array([[0.156, -0.010, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194],  # X transformation
                         [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048],  # Y transformation
                         [-0.022, -0.102, 0.229, 0.310, 0.246, 0.063, -0.055, -0.108]])  # Z transformation
    vcg = np.dot(invDower, ecg_toconvert)
    return vcg


def maximal_amplitude(vcg):
    """
    This function allows to find the point in the plane where the vcg amplitude is maximum.
    This will allow to compute the VECG Angle right after
    """
    time = np.argmax(np.linalg.norm(vcg[0:2], axis=0))
    maximal_coordinates = vcg[0][time], vcg[1][time]
    return maximal_coordinates


def VECGAng(ecg):
    """
    This function computes the VECG Angle, the feature we needed according to the paper:
    Automatic detection of premature atrial contractions in the electrocardiogram
    Vessela T. Krasteva, Irena I. Jekova, Ivaylo I. Christov
    """
    vcg = ecgtovcg(ecg)
    coordinates = maximal_amplitude(vcg)
    VECGang_max = np.angle(complex(coordinates[0], coordinates[1]), deg=True)
    if VECGang_max < 0:
        VECGang_max = 360 + VECGang_max
    return VECGang_max


def bsqi(refqrs, testqrs, agw=0.05, fs=500):
    """
    This function is based on the following paper:
        Li, Qiao, Roger G. Mark, and Gari D. Clifford.
        "Robust heart rate estimation from multiple asynchronous noisy sources
        using signal quality indices and a Kalman filter."
        Physiological measurement 29.1 (2007): 15.
    The implementation itself is based on:
        Behar, J., Oster, J., Li, Q., & Clifford, G. D. (2013).
        ECG signal quality during arrhythmia and its application to false alarm reduction.
        IEEE transactions on biomedical engineering, 60(6), 1660-1666.
    :param refqrs:  Annotation of the reference peak detector (Indices of the peaks).
    :param testqrs: Annotation of the test peak detector (Indices of the peaks).
    :param agw:     Agreement window size (in seconds)
    :param fs:      Sampling frquency [Hz]
    :returns F1:    The 'bsqi' score, between 0 and 1.
    """
    try:
        agw *= fs
        if len(refqrs) > 0 and len(testqrs) > 0:
            NB_REF = len(refqrs)
            NB_TEST = len(testqrs)

            tree = cKDTree(refqrs.reshape(-1, 1))
            Dist, IndMatch = tree.query(testqrs.reshape(-1, 1))
            IndMatchInWindow = IndMatch[Dist < agw]
            NB_MATCH_UNIQUE = len(np.unique(IndMatchInWindow))
            TP = NB_MATCH_UNIQUE
            FN = NB_REF - TP
            FP = NB_TEST - TP
            Se = TP / (TP + FN)
            PPV = TP / (FP + TP)
            if (Se + PPV) > 0:
                F1 = 2 * Se * PPV / (Se + PPV)
                _, ind_plop = np.unique(IndMatchInWindow, return_index=True)
                Dist_thres = np.where(Dist < agw)[0]
                meanDist = compute_mean(Dist[Dist_thres[ind_plop]]) / fs
            else:
                return 0
    except:
        F1 = 0
    else:
        F1 = 0
    return F1


def extraction_feature_final(ecg, freq, features_dict, lead):
    """This function produces the features for a 12-leads ecg from every pathology and concatenate them into a df"""
    features_global = pd.DataFrame()
    features_AF = extraction_feature_AF(ecg, freq, features_dict, lead)
    features_AVB = extraction_feature_AVB(ecg, freq, features_dict, lead)
    features_PAC = extraction_feature_PAC(ecg, freq, features_dict, lead)
    features_PVC = extraction_feature_PVC(ecg, freq, features_dict, lead)
    features_ST = extraction_feature_ST(ecg, freq, features_dict, lead)
    features_global = pd.concat([features_global, features_AF], axis=1)
    features_global = pd.concat([features_global, features_AVB], axis=1)
    features_global = pd.concat([features_global, features_PAC], axis=1)
    features_global = pd.concat([features_global, features_PVC], axis=1)
    features_global = pd.concat([features_global, features_ST], axis=1)
    return features_global



def frequency_regularity(data, lead, freq, eng):
    """
    This function computes the frequence deviation, in order to discriminate AF and AFL
    """
    try:
        ecg_lead = data[lead]
        ecgs = np.array_split(ecg_lead, 6)
        dominant_frequencies = list()
        for i, ecg_window in enumerate(ecgs):
            qrs = wfdb.processing.gqrs_detect(fs=500, sig=np.asarray(ecg_window.tolist() * 10), adc_gain=1000,
                                              adc_zero=0)
            max_freq = eng.f_wave_detection(matlab.double(ecg_window.tolist()), matlab.double(qrs.tolist()),
                                            matlab.double([500]))
            dominant_frequencies.append(max_freq)
        std = np.std(dominant_frequencies)
    except:
        std = 0
        print('We could not manage to get the std for the lead', lead)
    return std


def preprocessing_check(ecg):
    """
    This function allows you to select ecgs based on 2 criteria:
    - A signal quality threshold (median signal quality accross the leads). TODO later"Automatic diagnosis of the 12-lead ECG using a dee
    - The preprocessing check from Ribeiro, Antônio H., et al. p neural network."
    """
    Keep = (np.max(np.abs(np.array(ecg[0]) + np.array(ecg[2]) - np.array(ecg[1]))) == 0) & (
        np.max(np.abs(np.array(ecg[3]) + np.array(ecg[4]) + np.array(ecg[5])) == 0))
    return Keep


def get_features_from_QRS(index=0, preprocess=False):
    physionet_matlab.initialize_runtime([])
    matlab_runtime = physionet_matlab.initialize()
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    files_selected = pd.read_csv(testing_location)
    names_selected = files_selected['FileName'].values
    files = [name + '.mat' if not name.endswith('.mat') else name for name in names_selected]
    print(len(files), 'files to process')
    for i, f in enumerate(files):
        if i < index:
            continue
        else:
            #try:
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            ecg = []
            if f.startswith('S'):
                for lead in data:
                    seconds = int(len(lead) / 1000)
                    ecg.append(scipy.signal.resample(lead, int(seconds * 500)))
            for i_lead, lead in enumerate(list_lead):
                ecg_lead = data[lead]
                ecg_filtered = bandpass_filter(ecg_lead, 0.05, 100, 500, 3)
                ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 48, 52, 500, 3)
                ecg_filtered = ecg_filtered - bandpass_filter(ecg_filtered, 58, 62, 500, 3)
                ecg.append(ecg_filtered)
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
            DataFrame_sample_i = extraction_feature_wavedet(filename='', list_lead=list_lead, ecg=ecg,
                                                            freq=500, runtime=matlab_runtime)
            DataFrame_sample_i_BBB = extraction_feature_wavedet_BBB(filename='',
                                                                    ecg=ecg, list_lead=list_lead,
                                                                    freq=500, runtime=matlab_runtime)
            DataFrame_sample_i = pd.concat([DataFrame_sample_i, DataFrame_sample_i_BBB], axis=1)
            Ang = VECGAng(ecg)
            DataFrame_sample_i['VECGAng'] = Ang
            DataFrame_sample_i['Age'] = age
            DataFrame_sample_i['Sex'] = sex
            DataFrame_sample_i['Label'] = 'other'
            DataFrame_sample_i['Filename'] = f
            if i == 0:
                DataFrame_sample_i.to_csv(test, index=False, header=True)
            else:
                DataFrame_sample_i.to_csv(test, index=False, header=False, mode='a')
            print(i + 1, "signals processed for testing")


##############################################################
############### Machine Learning Part ########################
##############################################################


def split_data(features, test_size=0.2):
    label_old = features['Label']
    features = features.drop(['Label'], axis=1)
    le = LabelEncoder()
    label = le.fit_transform(label_old)
    return train_test_split(features, label, test_size=0.2, stratify=label)


def preprocessing_data(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def CrossFoldValidation(X_train, y_train, search=None):
    scoring_competition = make_scorer(compute_challenge_metrics_CV, greater_is_better=True)
    classifier, random_grid = RF_classifier()
    if search is not None:
        random_model = GridSearchCV(estimator=classifier, param_grid=random_grid, verbose=1,
                                    scoring=scoring_competition,
                                    n_jobs=-1, return_train_score=True, cv=6, refit=True)
    else:
        random_model = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, verbose=1,
                                          n_jobs=-1, scoring=scoring_competition, return_train_score=True, cv=6)
    random_model.fit(X_train, y_train)
    return random_model


def compute_challenge_metrics_CV(y_val, y_pred):
    if (len(y_pred)) > 0:
        num_classes = len(test_pathologies_ABE)
        # y_pred = model.predict(X_val)
        y_pred_beta = np.zeros((len(y_pred), num_classes))
        y_val_beta = np.zeros((len(y_pred), num_classes))
        for i in range(len(y_pred)):
            labels_pred = np.zeros(num_classes)
            labels_test = np.zeros(num_classes)
            labels_pred[y_pred[i]] = 1
            labels_test[y_val[i]] = 1
            y_pred_beta[i] = labels_pred
            y_val_beta[i] = labels_test
        final_metrics = compute_challenge_metric(weights, y_val_beta, y_pred_beta, test_pathologies_ABE,
                                                 normal)

        return final_metrics
    else:
        return 1


def TestSet(X_test, y_test, model, result_writing=None,
            plot_confusion=None):  # in order to allow multi label classification? no
    num_classes = len(test_pathologies_ABE)
    X_test = np.nan_to_num(X_test)
    y_pred_total_beta = model.predict_proba(X_test)  # size : (len(y_test), num_classes)
    for i in range(len(y_test)):
        classes = y_pred_total_beta[i] > 0.3  # todo: adapt the threshold
        y_pred_i = np.zeros(num_classes)
        if len(np.argwhere(classes)) > 0:
            for k in range(len(classes)):
                if classes[k]:
                    y_pred_i[k] = 1
        else:
            ind = np.argmax(y_pred_total_beta[i])
            y_pred_i[ind] = 1
        y_pred_total_beta[i] = y_pred_i
    y_test_beta = np.zeros((len(y_test), num_classes))
    for i in range(len(y_test)):
        labels_test = np.zeros(num_classes)
        labels_test[y_test[i]] = 1
        y_test_beta[i] = labels_test
    A = compute_confusion_matrices(y_test_beta, y_pred_total_beta)
    fbeta, gbeta, fbeta_measure, gbeta_measure = compute_beta_measures(y_test_beta, y_pred_total_beta, 2)
    fbeta_dict, gbeta_dict = dict(zip(pathologies_ABE, fbeta_measure)), dict(
        zip(pathologies_ABE, gbeta_measure))
    final_metrics = compute_challenge_metric(weights, y_test_beta, y_pred_total_beta, test_pathologies_ABE, normal)
    if result_writing is not None:
        y_test_classes = [pathologies_ABE[label] for label in y_test]
        with open(result_writing, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['The repartition of classes on the test set is, ', pd.Series(y_test_classes).value_counts()])
            for i, matrix in enumerate(A):
                writer.writerow(['Confusion matrix for the pathology', pathologies_ABE[i]])
                writer.writerow([matrix])
    if plot_confusion is not None:
        B = compute_modified_confusion_matrix(y_test_beta, y_pred_total_beta)
        f, axarr = plt.subplots(1, 2, figsize=(25, 20))
        axarr[0] = plt.subplot(211)
        axarr[0].matshow(B, cmap=plt.cm.Blues)
        for i in range(num_classes):
            for j in range(num_classes):
                c = B[j, i]
                axarr[0].text(i, j, str(c), va='center', ha='center')
        axarr[1] = plt.subplot(212)
        axarr[1].matshow(weights, cmap=plt.cm.Blues)
        for i in range(num_classes):
            for j in range(num_classes):
                c = weights[j, i]
                axarr[1].text(i, j, str(c), va='center', ha='center')
        f.savefig(plot_confusion)
    return fbeta, gbeta, final_metrics, fbeta_dict, gbeta_dict


def RF_classifier():
    # Number of trees in random forest
    n_estimators = [50, 90, 110, 130, 150, 200, 220, 250, 300]
    n_estimators_over = [50, 90, 110, 150, 200]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=5)]
    max_depth_over = [int(x) for x in np.linspace(10, 70, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 10, 15]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 4, 5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    weighted = ['balanced', 'balanced_subsample']
    # Create the random grid
    random_grid = {'n_estimators': n_estimators_over,
                   'max_features': max_features,
                   'max_depth': max_depth_over,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': weighted

                   }
    # Use the random grid to search for best hyperparameters

    rf = RandomForestClassifier()
    return rf, random_grid


def BuildClassifier(features, result_writing, write=True, search=None, results_plot_location=None):
    X_train, X_test, y_train, y_test = split_data(features)
    X_train, X_test = preprocessing_data(X_train, X_test)
    model_selected = CrossFoldValidation(X_train, y_train, search=None)
    dict_results = model_selected.cv_results_
    print(dict_results)
    index = np.argwhere(dict_results['rank_test_score'] == 1)[0][0]
    mean_score = dict_results['mean_test_score'][index]
    std_score = dict_results['std_test_score'][index] * 2
    f_beta, g_beta, competition, f_beta_dict, g_beta_dict = TestSet(X_test, y_test, model_selected, result_writing,
                                                                    results_plot_location)
    if write:
        with open(result_writing, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['The parameters tuned here are', model_selected.best_estimator_])
            writer.writerow(['Results we got with ', len(features.columns), 'features: ', features.columns])
            writer.writerow(['Results of the best model on validation set ', mean_score, '+/-', std_score])
            writer.writerow(['Mean F_beta result of the best model on test set ', f_beta])
            writer.writerow(['Mean G_beta result of the best model on test set ', g_beta])
            writer.writerow(['F_beta classes results of the best model on test set ', f_beta_dict])
            writer.writerow(['G_Beta classes results of the best model on test set ', g_beta_dict])
            writer.writerow(['Competition results of the best model on test set ', competition])
    return model_selected, f_beta


def extraction_features_test_set(index=0):
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    files_selected = pd.read_csv(index_test_set)
    names_selected = files_selected['FileName'].values
    files = [name + '.mat' if not name.endswith('.mat') else name for name in names_selected]
    corresponding_labels = files_selected['Label'].values
    corresponding_pathologies = np.squeeze([np.asarray(pathologies_all)[
                                                np.where(np.array(test_pathologies_all) == str(label))[0]] if str(
        label) in test_pathologies_all else np.nan for label in
                                            corresponding_labels])
    print(corresponding_pathologies)
    for i, f in enumerate(files):
        if i < index:
            continue
        if corresponding_pathologies[i] == 'SNR':
            continue
        else:
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            freq = get_freq(header_data)
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
            try:
                DataFrame_sample_i = extraction_feature_wavedet(filename='', list_lead=list_lead, ecg=ecg,
                                                                freq=int(freq), eng=eng)
                DataFrame_sample_i_BBB = extraction_feature_wavedet_BBB(filename='',
                                                                        ecg=ecg, list_lead=list_lead,
                                                                        freq=int(freq), eng=eng)
                DataFrame_sample_i = pd.concat([DataFrame_sample_i, DataFrame_sample_i_BBB], axis=1)
                Ang = VECGAng(ecg)
                DataFrame_sample_i['VECGAng'] = Ang
                DataFrame_sample_i['Age'] = age
                DataFrame_sample_i['Sex'] = sex
                DataFrame_sample_i['Label'] = corresponding_pathologies[i]
                DataFrame_sample_i['Filename'] = f
            except:

                print('We could not extract features from example', i + 1)
                continue
            if i == 0:
                DataFrame_sample_i.to_csv(test_set_location, index=False, header=True)
            else:
                DataFrame_sample_i.to_csv(test_set_location, index=False, header=False, mode='a')
        print(i + 1, "signals processed for testing")


# todo: étudie pourquoi on galère: preprocessing check experiment one
# todo: add max PR in the challenge, for LPR
if __name__ == '__main__':
    experiment_1 = 'wait'
    experiment_multi = 'done'
    experiment_extraction = ''
    results_experiments = {'done', 'wait', 'fail'}
    if experiment_1 not in results_experiments:
        """
        We are going to test which ecgs need to be removed from training based on the preprocessing check from Ribeiro
        """
        input_files = []
        remove_files = []
        i = 0
        for f in os.listdir(input_directory):
            
            if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith(
                    '.') and f.lower().endswith('mat'):
                input_files.append(f)
                i += 1
        print(i, "signals to process")
        input_files.sort()
        for i, f in enumerate(input_files):
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            if not preprocessing_check(data):
                remove_files.append(f)
        print(len(remove_files) / len(input_files))
        features_train = pd.read_csv('/home/david/Incremental_study/Data/features_train.csv')
        print(features_train)
        filenames = features_train['Filename']
        indexes_to_drop = []
        for i, filename in enumerate(filenames.values.tolist()):
            if filename in remove_files:
                indexes_to_drop.append(filenames.index.tolist()[i])
        print(indexes_to_drop)
        features_train.drop(indexes_to_drop, axis=0, inplace=True)
        print(features_train)
    if experiment_multi not in results_experiments:
        location_features = '/home/david/Incremental_study/Data/features_test_multilabelled.csv'
        to_be_changed = pd.read_csv(location_features)
        filenames = to_be_changed['Filename'].values
        filenames_test = []
        pathologies_test = []
        modif = []
        for i, f in enumerate(os.listdir(input_directory)):
            if f in filenames:
                tmp_input_file = os.path.join(input_directory, f)
                _, header_data = load_challenge_data(tmp_input_file)
                for iline in header_data:
                    if iline.startswith('#Dx'):
                        labels = iline.split(':')[1]
                        labels = labels.split(',')
                        pathologies = set([annot_to_patho.get(code.strip()) for code in labels if
                                           code.strip() in test_pathologies_all])
                        if len(pathologies) == 1 and pathologies.__contains__('SNR'):
                            modif.append(f)
                        pathologies_test.append(pathologies)
                        filenames_test.append(f)
        to_be_switched = random.sample(modif, 1800)
        indexes_to_switch = to_be_changed.index[to_be_changed['Filename'].apply(lambda x: x in to_be_switched)].tolist()
        features_to_switch = to_be_changed.iloc[indexes_to_switch]
        to_be_changed.drop(indexes_to_switch, axis=0, inplace=True)
        patho_final = []
        for patho in pathologies_test:
            for p in patho:
                patho_final.append(p)
        pathologies_test_encoded = []
        for i in range(len(pathologies_test)):
            sample_i = np.zeros(24)
            for patho in pathologies_test[i]:
                index = np.argwhere(np.asarray(pathologies_effective) == patho)[0]
                sample_i[index] = 1
            pathologies_test_encoded.append(sample_i)
        dict = (dict(zip(filenames_test, pathologies_test_encoded)))
        to_be_changed['Labels'] = to_be_changed['Filename'].apply(lambda x: dict.get(x))
        print(to_be_changed.columns)
        print(np.sum(to_be_changed['Labels'].values, axis=0))
        to_be_changed.to_csv(location_features)
    if experiment_extraction not in results_experiments:
        get_features_from_QRS()
