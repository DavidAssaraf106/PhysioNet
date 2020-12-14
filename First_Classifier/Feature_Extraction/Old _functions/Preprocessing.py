import os
import sys
import numpy as np
import matlab.engine
import pywt  # todo: pywt ne fonctionne pas
from scipy.io import loadmat
# from skimage.restoration import denoise_wavelet
from Feature_Extraction import *
from Preprocessing_features import *



def preprocess_None(data, list_lead, parameters):
    return data

def preprocess_notch(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        filtered_ecg_measurements = bandpass_filter(ecg_lead, lowcut=48, highcut=52,
                                                    signal_freq=500, filter_order=2)
        notch_ecg_measurements = ecg_lead - filtered_ecg_measurements
        data_bis[num_lead] = notch_ecg_measurements

    return data_bis


'''
The preprocessing drift method is directly implemented from the paper 'Automatic detection of premature atrial contractions
in the electrocardiogram, Vessela T. Krasteva, Irena I. Jekova, Ivaylo I. Christov'.

'''
def preprocess_drift(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        filtered_ecg_measurements = bandpass_filter(ecg_lead, lowcut=2.2, highcut=200,
                                                    signal_freq=500, filter_order=3)

        data_bis[num_lead] = filtered_ecg_measurements

    return data_bis


def preprocess_bw(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        new_ecg_measurement = ecg_lead
        data_bis[num_lead] = new_ecg_measurement

    return data_bis

def preprocess_wavelet(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        new_ecg_measurement = ecg_lead
        data_bis[num_lead] = new_ecg_measurement

    return data_bis

def preprocess_filter_bandpass(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        new_ecg_measurement = ecg_lead
        data_bis[num_lead] = new_ecg_measurement

    return data_bis

def preprocess_padding(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        new_ecg_measurement = ecg_lead
        data_bis[num_lead] = new_ecg_measurement

    return data_bis

def preprocess_mirroring(data, list_lead, parameters):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        new_ecg_measurement = ecg_lead
        data_bis[num_lead] = new_ecg_measurement

    return data_bis