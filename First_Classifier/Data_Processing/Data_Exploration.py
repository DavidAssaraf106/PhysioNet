"""
In this file, we are going to explore the quality and the characteristics of the new data released by the competition.
We are going to perform a general study of these datasets, and also a signal quality exploration in order to assess their quality
The origins of these datasets are multiple:
- CPSC 2018: Approximately 10k 12-leads ECG from the same origins: 11 hospitals in China.
Location: C:\\Users\\David\\PhysioNet_Code\\Training_WFDB
- INCART 12-leads Arruthmia DataBase, St-Petersburg: 74 recordings (a priori long)
Location: C:\\Users\\David\\PhysioNet_Code\\Training_StPetersburg
- PTB Diagnostic 12-leads ECG Database: 600 recordings
Location: C:\\Users\\David\\PhysioNet_Code\\Training_PTB
- PTB-XL Diagnostic 12-leads ECG Database: 22k recordings
Location: C:\\Users\\David\\PhysioNet_Code\\Training_StPetersburg\\Training_PTB_XL
- Georgia 12-Lead ECG Challenge Database: 10k recordings
Location: C:\\Users\\David\\PhysioNet_Code\\Training_StPetersburg\\Training_Georgia
"""

import warnings
import csv
import matlab.engine
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import wfdb.processing
from scipy.signal import butter, lfilter
import os
import matplotlib.mlab as mlab
import math
import os
import pandas as pd
import sys
from scipy.io import loadmat
from scipy.signal import savgol_filter, butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

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
                  '713427006': 'CRBBB',
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
                  '164884008': 'PVC',
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
                  '63593006': 'SVPB',
                  '426761007': 'SVT',
                  '251139008': 'ALR',
                  '164934002': 'TAb',
                  '59931005': 'TInv',
                  '251242005': 'UTall',
                  '164937009': 'UAb',
                  '11157007': 'VBig',
                  '17338001': 'VEB',
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
                  '74390002': 'WPW'}


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def get_freq(header_data):
    for lines in header_data:
        tmp = lines.split(' ')
        return tmp[2]


def explore_data(input_directory):
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
    input_files.sort()
    DataFrame_results = pd.DataFrame()
    for i, f in enumerate(input_files):
        DataFrame_i = pd.DataFrame()
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        freq = int(get_freq(header_data))
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
                label = iline.split(': ')[1].split(',')[0].strip()
        index_quality = bsqi(wfdb.processing.gqrs_detect(fs=freq, sig=data[0], adc_gain=1000, adc_zero=0),
                             wfdb.processing.xqrs_detect(sig=data[0], fs=freq, verbose=False))
        DataFrame_i['freq'] = [freq]
        DataFrame_i['quality'] = [index_quality]
        DataFrame_i['len'] = [len(data[0])]
        DataFrame_i['Age'] = [age]
        DataFrame_i['Sex'] = [sex]
        DataFrame_i['Label'] = [label]
        DataFrame_results = pd.concat([DataFrame_results, DataFrame_i], axis=0)
    return DataFrame_results


def conversion_label(int_label):
    return annot_to_patho[str(int_label)]


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
        else:
            return 0

    else:
        F1 = 0
    return F1


if __name__ == '__main__':
    experiment_1 = 'done'
    experiment_2 = ''
    results_experiments = {'done', 'fail'}
    if experiment_1 not in results_experiments:
        """
        Create a DataFrame containing the information of the several databases for every database
        China: C:\\Users\\David\\PhysioNet_Code\\Training_WFDB
        Russia: C:\\Users\\David\\PhysioNet_Code\\Training_StPetersburg
        PTB: C:\\Users\\David\\PhysioNet_Code\\Training_PTB
        PTB_XL: C:\\Users\\David\\PhysioNet_Code\\Training_PTB_XL
        Georgia: C:\\Users\\David\\PhysioNet_Code\\Training_StPetersburg\\Training_Georgia
        """
        DataFrame_CPSC = explore_data('C:\\Users\\David\\PhysioNet_Code\\Training_WFDB')
        DataFrame_CPSC.to_csv('C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\New_Data\\Explo_CPSC.csv',
                              index=False, header=True)
    if experiment_2 not in results_experiments:
        """
        Analyzing the exploration of the new CPSC database
        CSV location: C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\New_Data\\Explo_CPSC.csv
        Results location: C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\New_Data\\Features_Explo_CPSC.csv
        """
        features_to_be_analyzed = pd.read_csv(
            'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\New_Data\\Explo_CPSC.csv')
        features_to_be_analyzed.describe().to_csv(
            'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\New_Data\\Features_Explo_CPSC.csv')
        features_to_be_analyzed['Label'] = features_to_be_analyzed['Label'].apply(lambda x: annot_to_patho[str(x)])
        print(features_to_be_analyzed['Label'].value_counts())
