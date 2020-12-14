import os
import sys

import matlab.engine
import pywt
from scipy.io import loadmat
# from skimage.restoration import denoise_wavelet
from Feature_Extraction import *
from Preprocessing_features import *


# insert your input directory and the matlab paths
input_directory = "C:\\Users\\David\\PhysioNet_Code\\Training_WFDB"
matlab_path1 = "C:\\Users\\David\\ecg-kit-master\\common\\wavedet"
matlab_path2 = "C:\\Users\\David\\ecg-kit-master\\common"
data_extracted = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils"

lead_to_func_ending_dict = {0: "I", 1: "II", 2: "III", 3: "aVR", 4: "aVL", 5: "aVF", 6: "V1", 7: "V2",
                            8: "V3",
                            9: "V4", 10: "V5", 11: "V6"}

interesting_keys = ["Pon", "P", "Poff", "QRSon", "Q", "R", "S", "QRSoff", "Ton", "T", "Toff"]


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


# TODO: integration and denoising of the data is useful?
# TODO: power-line interference, tremor noise and base-line
# TODO: drift distortions: frequence for the bandpass filter
# TODO: notch filter, low-pass filter, high-pass recursive filter
# TODO: what is the expected size of A0001?
# todo: try wavelet with pywt

def preprocess_filter_pywt(data, list_lead):
    data_bis = data.copy()  # .tolist()?
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        (ca, cd) = pywt.dwt(ecg_lead, 'haar')
        ecg_lead_rec = pywt.idwt(ca, cd, 'haar')
        data_bis[num_lead] = ecg_lead_rec
    return data_bis
#todo : low cutoff frequency? 
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
        filtered_ecg_measurements = denoise_wavelet(filtered_ecg_measurements/10000, wavelet='db4', sigma=0.008, rescale_sigma=True)*10000
        filtered_ecg_measurements = bandpass_filter(data[num_lead], lowcut=0.01, highcut=200,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=150,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=80,
                                                    signal_freq=500, filter_order=3)  # 120 était good
        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
        data_bis[num_lead] = filtered_ecg_measurements
    return data_bis


def preprocess_filter_PT(data, list_lead):
    data_bis = data.copy()  # .tolist()?
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        filtered_ecg_measurements = bandpass_filter(data[num_lead], lowcut=0.5, highcut=200,
                                                    signal_freq=500, filter_order=3)

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
        # filtered_ecg_measurements = denoise_wavelet(filtered_ecg_measurements/1000, wavelet='db4', rescale_sigma=True)*1000
        # filtered_ecg_measurements = integrate(filtered_ecg_measurements, int(500 / 70))
        data_bis[num_lead] = filtered_ecg_measurements
    return data_bis


def preprocess_filter_ECG(data, list_lead):
    data_bis = data.copy()
    for i in range(len(list_lead)):
        num_lead = list_lead[i]
        ecg_lead = data_bis[num_lead]
        filtered_ecg_measurements = denoise_wavelet(data[num_lead] / 10000, wavelet='db4', sigma=0.008,
                                                    rescale_sigma=True) * 10000
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=200,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=150,
                                                    signal_freq=500, filter_order=3)
        filtered_ecg_measurements = bandpass_filter(filtered_ecg_measurements, lowcut=0.01, highcut=150,
                                                    signal_freq=500, filter_order=3)
        # print(filtered_ecg_measurements - data[num_lead])
        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
        data_bis[num_lead] = filtered_ecg_measurements

    return data_bis


# we choose our own Q and S points extracted from the R-peaks since the wavedet destroys too much values
# TODO: check the Q and S points output by the wavedet approach
# TODO: why is the first signal smooth and not the others?
def extraction_feature_wavedet(filename, list_lead, ecg, freq, eng, qrs_peak=None):
    wavedet_3D_dict = wavedet_3D_wrapper(filename=filename, list_lead=list_lead, qrs_peak=qrs_peak, eng=eng, ecg=ecg)
    signal_DataFrame = pd.DataFrame()
    for i, lead in enumerate(list_lead):
        wavedet_3D_dict_i = wavedet_3D_dict[i]
        extracted_features = [(key, np.asarray(wavedet_3D_dict_i[key], dtype=np.int32)) for key in interesting_keys]
        dict_features = {key: value for key, value in extracted_features}
        thismodule = sys.modules[__name__]
        curr_extarction_func = getattr(thismodule, "extraction_feature_{}".format(lead_to_func_ending_dict[lead]))
        feat_lead = curr_extarction_func(ecg[lead], freq, dict_features)
        signal_DataFrame = pd.concat([signal_DataFrame, feat_lead], sort=False, axis=1)
    return signal_DataFrame


def wavedet_3D_wrapper(filename, list_lead, qrs_peak, eng, ecg):  # engine does not support structure array
    ecg_matlab = []
    for i in range(len(ecg)):
        ecg_matlab.append(ecg[i].tolist())
    if qrs_peak == []:
        ret_val = eng.python_wrap_wavedet_3D(input_directory, matlab.double(ecg_matlab), filename, list_lead, [], nargout=6)
    else:
        ret_val = eng.python_wrap_wavedet_3D(input_directory, matlab.double(ecg_matlab), filename, list_lead, qrs_peak, nargout=6)
    for i, lead in enumerate(list_lead):
        # TODO: change to another qrs detector when such a situation occurs and see if it works better
        if qrs_peak == []:
            qrs_peak_i = np.array(ret_val[i]["R"]._data).reshape(ret_val[i]["R"].size, order='F')[0]
            qrs_peak_i[np.isnan(qrs_peak_i)] = 0
            qrs_peak_i = np.asarray(qrs_peak_i, dtype=np.int64)
        else:
            qrs_peak_i = np.asarray(qrs_peak[i], dtype=np.int64)
        if len(qrs_peak_i[qrs_peak_i > 0]) > 5:
            for key in (ret_val[i].keys()):
                if interesting_keys.__contains__(key):
                    ret_val[i][key] = np.array(ret_val[i][key]._data).reshape(ret_val[i][key].size, order='F')[0]
                    ret_val[i][key][np.isnan(ret_val[i][key])] = 0
                    ret_val[i][key] = np.asarray(ret_val[i][key][1:], dtype=np.int64)
            R_ref = ret_val[i]['R']
            Poff = ret_val[i]['Poff']
            Ton = ret_val[i]['Ton']
            for p, ind in enumerate(ret_val[i]['QRSon']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['Q'][p] == 0:
                        if R_ref[p] > ind+1:
                            candidate = np.argmin(ecg_lead[ind:R_ref[p]]) + ind  # todo: à remplacer par compute_argmin?
                            if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                ref_value = np.max(np.abs(ecg_lead[ind:ind+5]))

                                if Poff[p] > 0:
                                    indice_minimal = (candidate + Poff[p])/2
                                else:
                                    indice_minimal = 0
                                while ind > 0 and np.abs((ecg_lead[ind] - ref_value) / ref_value) < 0.2 and ind > indice_minimal: #todo: divide by zero encountered
                                    ind = ind - 1
                                ret_val[i]['QRSon'][p] = ind
                            ret_val[i]['Q'][p] = candidate  # in order to correct when we do not detect any Q points
                        else:
                            continue
            for p, ind in enumerate(ret_val[i]['QRSoff']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['S'][p] == 0:
                        if ind > R_ref[p]+1:
                            candidate = np.argmin(ecg_lead[R_ref[p]:ind]) + R_ref[p]# todo: à remplacer par compute_argmin?
                            if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                ref_value = np.max(np.abs(ecg_lead[ind-5:ind]))

                                if Ton[p] > 0:
                                    indice_maximal = (candidate + Ton[p])/2
                                else:
                                    indice_maximal=len(ecg_lead)
                                while ind < len(ecg_lead) and (np.abs((ecg_lead[ind] - ref_value) / ref_value)) and ind < indice_maximal:
                                    ind = ind + 1
                                ret_val[i]['QRSoff'][p] = ind
                            ret_val[i]['S'][p] = candidate  # in order to correct when we do not detect any S points
                        else:
                            continue

        else:
            for key in (ret_val[i].keys()):
                ret_val[i][key] = np.asarray([0])
                print("We are having troubles with", filename + ' lead ', lead)
                plot_QRS(ecg[i], qrs_peak_i, lead, filename)

    return ret_val


# TODO: check if the QRS detection si robust across the leads of a same sample (for the leads in which we should have)
# gqrs = True: you want to use the wavedet algorithm with the wfdb qrs detector gqrs

def get_features_from_QRS(model=None, preprocess=True):
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

    list_lead = [0, 1, 6, 7, 8, 11]  # leads I, II, V1, V2, V3 and V6
    for i, f in enumerate(input_files):

        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        freq = get_freq(header_data)

        if preprocess:
            ecg = preprocess_filter_wavelet(data, list_lead)
        else:
            ecg = data

        gain_lead = get_gain_lead(header_data)

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
        if model is not None:
            R_matrix = []
            for p in range(len(list_lead)):
                thismodule = sys.modules[__name__]
                QRS_detector = getattr(thismodule, "QRS_detector_{}".format(model))
                R_points = QRS_detector(ecg, list_lead[p])
                R_matrix.append(R_points.tolist())
            reference_length = np.min([len(R_matrix[k]) for k in range(len(R_matrix))])
            R_matrix = [R_matrix[j][:reference_length] for j in range(len(R_matrix))]
            DataFrame_sample_i = extraction_feature_wavedet(filename=input_files[i], list_lead=list_lead, ecg=ecg,
                                                            freq=freq,
                                                            eng=eng, qrs_peak=matlab.double(R_matrix))

        else:
            DataFrame_sample_i = extraction_feature_wavedet(filename=input_files[i], list_lead=list_lead, ecg=ecg,
                                                            freq=freq, eng=eng)

        DataFrame_sample_i['Age'] = age
        DataFrame_sample_i['Sex'] = sex
        DataFrame_sample_i['Label'] = label
        if i == 0:
            DataFrame_sample_i.to_csv(data_extracted + "\\features_test_matlab.csv", index=False, header=True)
        else:
            DataFrame_sample_i.to_csv(data_extracted + "\\features_test_matlab.csv", index=False, header=False,
                                      mode='a')
        print(i+1, "signals processed")
    eng.quit()


# model in gqrs, xqrs, PT
if __name__ == '__main__':
    get_features_from_QRS(preprocess=False)
