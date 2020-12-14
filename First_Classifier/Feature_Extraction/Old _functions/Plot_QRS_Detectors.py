import os
import sys

import matlab.engine
from scipy.io import loadmat
from skimage.restoration import denoise_wavelet
from Feature_Extraction import *
from Preprocessing_features import *
from QRS_Detector_ECG_matlab import load_challenge_data, preprocess_filter_ECG, wavedet_3D_wrapper, \
    preprocess_filter_PT, preprocess_filter_wavelet, preprocess_filter_pywt
from QRSDetectors import QRS_detector_gqrs, QRS_detector_xqrs, QRS_detector_PT

# insert your input directory and the matlab paths
input_directory = "C:\\Users\\David\\PhysioNet_Code\\Training_WFDB"
matlab_path1 = "C:\\Users\\David\\ecg-kit-master\\common\\wavedet"
matlab_path2 = "C:\\Users\\David\\ecg-kit-master\\common"
path_to_images = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Preprocess_ECG"
path_to_morphological_features = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Morphological_ECG"

interesting_keys = ["Pon", "P", "Poff", "QRSon", "Q", "R", "S", "QRSoff", "Ton", "T", "Toff"]
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'fuchsia', 'lime', 'mediumpurple']


# TODO: we are with the PT preprocessing
# todo: why are the initial R-peaks so far from ground truth?
# todo: why are the performances so low on filtered signal?
# TODO: les techniques utilisées maintenant déforment qd mm pas mal les valeurs, peut-être peut-on s'intéresser à l'espérance de la différence
def extraction_feature_wavedet(filename, list_lead, model, ecg, eng, preprocessing, qrs_peak=[], plot=True,
                               comparison=True):
    wavedet_3D_dict = wavedet_3D_wrapper(filename=filename, list_lead=list_lead, qrs_peak=qrs_peak, eng=eng, ecg=ecg)
    thismodule = sys.modules[__name__]
    preprocessing_method = getattr(thismodule, "preprocess_filter_{}".format(preprocessing))
    if comparison:

        if model is not None:
            R_matrix = []
            QRS_detector = getattr(thismodule, "QRS_detector_{}".format(model))
            for p in range(len(list_lead)):
                R_points = QRS_detector(preprocessing_method(ecg, list_lead), list_lead[p])
                R_matrix.append(R_points.tolist())
            reference_length = np.min([len(R_matrix[k]) for k in range(len(R_matrix))])
            R_matrix = [R_matrix[j][:reference_length] for j in range(len(R_matrix))]
            wavedet_3D_dict_filtered = wavedet_3D_wrapper(filename=filename, list_lead=list_lead,
                                                          qrs_peak=matlab.double(R_matrix),
                                                          eng=eng, ecg=preprocessing_method(ecg, list_lead))
        else:
            wavedet_3D_dict_filtered = wavedet_3D_wrapper(filename=filename, list_lead=list_lead,
                                                          qrs_peak=qrs_peak,
                                                          eng=eng, ecg=preprocessing_method(ecg, list_lead))
    signal_DataFrame = pd.DataFrame()
    for i, lead in enumerate(list_lead):
        wavedet_3D_dict_i = wavedet_3D_dict[i]
        extracted_features = [np.asarray(wavedet_3D_dict_i[key][1:], dtype=np.int32) for key in
                              interesting_keys]
        if plot:
            if model is None:
                model_bis = 'default'
            else:
                model_bis = model
            images_file = path_to_morphological_features + '\\' + model_bis + '\\' + filename
            ecg_file = images_file + '\\lead_' + str(lead) + '.png'
            if os.path.exists(ecg_file):
                continue
            else:
                if not os.path.exists(images_file):
                    os.mkdir(images_file)
                if comparison:
                    wavedet_3D_dict_filtered_i = wavedet_3D_dict_filtered[i]
                    extracted_features_filtered = [np.asarray(wavedet_3D_dict_filtered_i[key], dtype=np.int32) for key
                                                   in interesting_keys]
                    # print({key: wavedet_3D_dict_i[key][:len(wavedet_3D_dict_filtered_i[key])]
                    # - wavedet_3D_dict_filtered_i[key][:len(wavedet_3D_dict_i[key])] for key in interesting_keys})

                    f, axarr = plt.subplots(1, 2, figsize=(25, 20))
                    axarr[0] = plot_morphological_features(ecg, lead, extracted_features, filename, model, 211,
                                                           qrs_peak)
                    if model is not None:
                        axarr[1] = plot_morphological_features(preprocessing_method(ecg, list_lead), lead,
                                                               extracted_features_filtered, filename,
                                                               model, 212, R_matrix)
                    else:
                        axarr[1] = plot_morphological_features(preprocessing_method(ecg, list_lead), lead,
                                                               extracted_features_filtered,
                                                               filename,
                                                               model, 212, [])
                    f.savefig(ecg_file)
                    plt.cla()
                    plt.clf()
                    plt.close()
                else:
                    fig = plot_morphological_features(ecg, lead, extracted_features, filename, model, qrs_peak)
                    fig.figure(figsize=(25, 10))
                    fig.savefig(ecg_file)
                    plt.cla()
                    plt.clf()
                    plt.close()

    return signal_DataFrame


# TODO: handle with the model=None for the titles
def plot_morphological_features(ecg, lead, extracted_features, filename, model=None, number_subplot=0,
                                qrs_initial=[]):  # indexes (0,1000)
    list_lead = [0, 1, 6, 7, 8, 11]
    for i in range(len(extracted_features)):
        extracted_features[i] = extracted_features[i][extracted_features[i] > 0]
        extracted_features[i] = extracted_features[i][extracted_features[i] < 3000]
    if len(ecg) == 12:
        ecg_plot = ecg[lead]
    else:
        ecg_plot = ecg
    if number_subplot > 0:
        axes = plt.subplot(number_subplot)
    axes.set_ylim(-2000, 2000)
    axes.plot(ecg_plot[:3000])
    if len(extracted_features) > 0:
        for i in range(len(interesting_keys)):
            if len(extracted_features[i]) > 0:
                axes.plot(extracted_features[i], ecg_plot[extracted_features[i]], marker='o', ls='', color=colors[i],
                          label=interesting_keys[i])
            else:
                continue
        axes.legend()
    if len(qrs_initial) > 0:
        ind_lead = list_lead.index(lead)
        qrs_initial = np.asarray(qrs_initial, dtype=np.int64)
        qrs_initial = qrs_initial[qrs_initial < 3000]
        axes.plot(qrs_initial[ind_lead], ecg_plot[qrs_initial[ind_lead]], marker='o', ls='', color='lightpink',
                  label='initial R_points')
        axes.legend()
    name = filename.split('.')[0]
    if model is not None:
        name = model + ' ' + name
    if number_subplot == 311 or number_subplot == 211:
        axes.set_title('Raw ECG ' + name + '_lead' + str(lead))
    if number_subplot == 312 or number_subplot == 212:
        axes.set_title('Filtered ECG ' + name + '_lead' + str(lead))
    if number_subplot == 313:
        axes.set_title('Difference between ECGs ' + name + '_lead' + str(lead))
    if number_subplot == 0:
        axes.set_title('Raw ECG ' + name + '_lead' + str(lead))

    return axes


# gqrs = True: you want to use the wavedet algorithm with the wfdb qrs detector gqrs
# TODO: pistes d'amélioration: combiner les anciens et nouveaux R-peaks.
# TODO: observer le Cpsc2018 et voir ce que ça donne quand on fait leur preprocessing
# TODO: pq est-ce qu'il ya  autant de diff entre les R-peaks initialement détectés et ceux du plot post-wavedet?
# TODO: insert verbose
def get_features_from_QRS(preprocessing, model=None, plot_preprocess=True):
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_path1, nargout=0)
    eng.addpath(matlab_path2, nargout=0)
    input_files = []
    thismodule = sys.modules[__name__]
    if model is not None:
        QRS_detector = getattr(thismodule, "QRS_detector_{}".format(model))
    preprocessing_method = getattr(thismodule, "preprocess_filter_{}".format(preprocessing))
    i = 0
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
            i += 1
    print(i, "signals to process")

    list_lead = [0, 1, 6, 7, 8, 11]  # leads I, II, V1, V2, V3 and V6
    for i, f in enumerate(input_files):
        if i < 0 :
            continue
        else:
            tmp_input_file = os.path.join(input_directory, f)
            data, header_data = load_challenge_data(tmp_input_file)
            ecg = data

            if plot_preprocess:
                directory_file = path_to_images + '\\' + preprocessing + '\\' + input_files[i]

                for k in range(len(list_lead)):
                    num_lead = list_lead[k]
                    ecg_file = directory_file + '\\lead_' + str(num_lead) + '.png'
                    if os.path.exists(ecg_file):
                        continue
                    else:
                        if not os.path.exists(directory_file):
                            os.mkdir(directory_file)
                    f, axarr = plt.subplots(1, 3, figsize=(25, 10))

                    axarr[0] = plot_morphological_features(data, num_lead, [], input_files[i], model, 311)
                    axarr[1] = plot_morphological_features(preprocessing_method(data, [num_lead]), num_lead, [],
                                                           input_files[i], model, 312)
                    axarr[2] = plot_morphological_features(
                        preprocessing_method(data, [num_lead])[num_lead] + data[num_lead], num_lead, [], input_files[i],
                        model, 313)
                    f.savefig(ecg_file)
                    plt.cla()
                    plt.clf()
                    plt.close()
            if model is not None:
                R_matrix = []
                for p in range(len(list_lead)):
                    R_points = QRS_detector(ecg, list_lead[p])
                    R_matrix.append(R_points.tolist())
                reference_length = np.min([len(R_matrix[k]) for k in range(len(R_matrix))])
                R_matrix = [R_matrix[j][:reference_length] for j in range(len(R_matrix))]
                extraction_feature_wavedet(filename=input_files[i], list_lead=list_lead, model=model, ecg=ecg,
                                           eng=eng, preprocessing=preprocessing, qrs_peak=matlab.double(R_matrix))

            else:

                extraction_feature_wavedet(filename=input_files[i], list_lead=list_lead, model=model, ecg=ecg, eng=eng,
                                           preprocessing=preprocessing)
            print(i + 1, "signals processed")
    eng.quit()


# preprocessing in {pywt, wavelet, ECG, PT}
if __name__ == '__main__':
    get_features_from_QRS(preprocessing='wavelet')
