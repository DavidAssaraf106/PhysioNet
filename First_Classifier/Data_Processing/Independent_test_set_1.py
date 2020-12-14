"""
This file will allow tou to:
- load the Independent test Set from Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al.
Automatic diagnosis of the 12-lead ECG using a deep neural network. Nat Commun 11, 1760 (2020).
- resample it from 400 to 500 Hz
- load the gold-standard annotations from this dataset
- load the meta-attributes for this dataset
"""

import numpy as np
import pandas as pd
import csv
import os
import h5py
import scipy.signal
import argparse
import matplotlib.pyplot as plt
from Global_Experiment_Single_Label import extraction_feature_wavedet_BBB, extraction_feature_final, extraction_feature_wavedet, VECGAng, split_data, preprocessing_data
import matlab.engine
import joblib
from Scoring_file import compute_modified_confusion_matrix, compute_confusion_matrices, compute_beta_measures, compute_challenge_metric, load_weights

pathologies_A = ['IAVB', 'RBBB', 'LBBB', 'AF']
pathologies = ['IAVB', 'RBBB', 'LBBB', 'SB', 'AF', 'SNR', 'ST']
test_pathologies = ['270492004', '59118001', '164909002', '426177001', '164889003', '426783006', '427084000']


normal = '426783006'

if os.name == 'posix':
    data_location = '/home/david/Independent_test_set_1/ecg_tracings.hdf5'
    annotation_location = '/home/david/Independent_test_set_1/gold_standard.csv'
    meta_data_location = '/home/david/Independent_test_set_1/attributes.csv'
    matlab_path1 = "/home/david/ecg-kit-master/common/wavedet"
    matlab_path2 = "/home/david/ecg-kit-master/common"
    test_set_location = '/home/david/Independent_test_set_1/features_1360.csv'
    weights_location = '/home/david/weights.csv'

if os.name == 'nt':
    data_location = 'C:\\Users\\David\\PhysioNet_Code\\Independent_Test_Set\\Independent_test_set_1\\ecg_tracings.hdf5'
    annotation_location = 'C:\\Users\\David\\PhysioNet_Code\\Independent_Test_Set\\Independent_test_set_1\\gold_standard.csv'
    meta_data_location = 'C:\\Users\\David\\PhysioNet_Code\\Independent_Test_Set\\Independent_test_set_1\\attributes.csv'
    matlab_path1 = "C:\\Users\\David\\ecg-kit-master\\common\\wavedet"
    matlab_path2 = "C:\\Users\\David\\ecg-kit-master\\common"
    test_set_location = 'C:\\Users\\David\\PhysioNet_Code\\Independent_Test_Set\\Independent_test_set_1\\features_test.csv'
    weights_location = 'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\weights.csv'



def extraction_features(data, annot, meta_data, index=0):
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_path1, nargout=0)
    eng.addpath(matlab_path2, nargout=0)
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print(len(data), 'signals to process for Independent testing')
    for i, ecg in enumerate(data):
        if i < index:
            continue
        try:
            DataFrame_sample_i = extraction_feature_wavedet(filename='', list_lead=list_lead, ecg=ecg,
                                                            freq=500, eng=eng)
            DataFrame_sample_i_BBB = extraction_feature_wavedet_BBB(filename='',
                                                                    ecg=ecg, list_lead=list_lead,
                                                                    freq=500, eng=eng)
            DataFrame_sample_i = pd.concat([DataFrame_sample_i, DataFrame_sample_i_BBB], axis=1)
            Ang = VECGAng(ecg)
            DataFrame_sample_i['VECGAng'] = Ang
            DataFrame_sample_i['Age'] = meta_data['age'].loc[i]
            DataFrame_sample_i['Sex'] = 0 if meta_data['sex'].loc[i].strip() == 'M' else 1
            DataFrame_sample_i['Label'] = annot[i]
            DataFrame_sample_i['Filename']  = f
            if i == 0:
                DataFrame_sample_i.to_csv(test_set_location, index=False, header=True)
            else:
                DataFrame_sample_i.to_csv(test_set_location, index=False, header=False, mode='a')
        except:
            print('We could not extract features for example', i+1)
        print(i + 1, "signals processed for testing")


def TestSet_short(y_pred_total_model, y_test_beta, results_writing, plot_confusion):
    label_to_patho_here = {'RBBB': 1, 'AF': 4, 'LBBB': 2, 'IAVB': 0,  'SB': 3, 'ST': 6}
    pathologies_ordered = sorted(label_to_patho_here, key=label_to_patho_here.get)
    A = compute_confusion_matrices(y_test_beta, y_pred_total_model)
    fbeta, gbeta, fbeta_measure, gbeta_measure = compute_beta_measures(y_test_beta, y_pred_total_model, 2)
    fbeta_dict, gbeta_dict = dict(zip(pathologies_ordered, fbeta_measure)), dict(
        zip(pathologies_ordered, gbeta_measure))
    final_metrics = compute_challenge_metric(weights, y_test_beta, y_pred_total_model, test_pathologies, normal)
    y_test_classes = [k for k, v in label_to_patho_here.items() for i in range(len(y_test_beta)) if v == y_test_beta[i]]
    with open(results_writing, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['The repartition of classes on the test set is, ', pd.Series(y_test_classes).value_counts()])
        for i, matrix in enumerate(A):
            writer.writerow(['Confusion matrix for the pathology', pathologies_ordered[i]])
            writer.writerow([matrix])
        writer.writerow(['The beta measures we had on the test set are:'])
        writer.writerow(['F_beta:', fbeta_dict])
        writer.writerow(['G_beta', gbeta_dict])
        writer.writerow(['The final classification metrics:', final_metrics])
    B = compute_modified_confusion_matrix(y_test_beta, y_pred_total_model)
    f, axarr = plt.subplots(1, 2, figsize=(25, 20))
    axarr[0] = plt.subplot(211)
    axarr[0].matshow(B, cmap=plt.cm.Blues)
    for i in range(len(test_pathologies)):
        for j in range(len(test_pathologies)):
            c = B[j, i]
            axarr[0].text(i, j, str(c), va='center', ha='center')
    axarr[1] = plt.subplot(212)
    axarr[1].matshow(weights, cmap=plt.cm.Blues)
    for i in range(len(test_pathologies)):
        for j in range(len(test_pathologies)):
            c = weights[j, i]
            axarr[1].text(i, j, str(c), va='center', ha='center')
    f.savefig(plot_confusion)
    return fbeta, gbeta, final_metrics, fbeta_dict, gbeta_dict


def reformat(y_output):
    num_classes = len(y_output[0])
    if num_classes == 14:
        indexes_to_keep = [0, 4, 6, 9, 10, 11, 12]
    if num_classes == 26:
        indexes_to_keep = [0, 4, 8, 18, 20, 21, 22]
    y_output = y_output[:, indexes_to_keep]
    y_right = y_output[:, [1, 3, 2, 4, 0, 6, 5]]
    return y_right


if __name__ == '__main__':
    experiment_0 = ''
    experiment_1 = 'wait'
    results_experiments = {'wait', 'done', 'fail'}
    if experiment_0 not in results_experiments:
        ecg_data = []
        with h5py.File(data_location, "r") as f:
            x = np.array(f['tracings'])
        for i, example in enumerate(x):
            ecg = []
            for lead in np.transpose(example):
                seconds = int(len(lead) / 400)
                # ecg.append(scipy.signal.resample(lead, int(seconds * 500)))
                ecg.append(lead)
            ecg_data.append(ecg)
        ecg_data = np.asarray(ecg_data)
        annotations = []
        with open(annotation_location, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                array = [pathologies[index] for index, label in enumerate(row[0].split(',')) if label == '1']
                if len(array) == 0:  # 'other' pathology
                    array = ['other']
                annotations.append(array[:1])
        meta_data = pd.read_csv(meta_data_location)
        print(np.squeeze(np.asarray(annotations)))
        print(pd.Series(np.squeeze(np.asarray(annotations))).value_counts())
        extraction_features(ecg_data, annotations, meta_data)
    if experiment_1 not in results_experiments:
        """
        Here, we are going to test the generalization ability of our model when being trained on DataBases A and B.
        Later, I wish to see how well they perform when being trained solely on DataBase A (requires to change the pathologies 
        that we consider).
        """
        location_model = ''
        model = joblib.load(location_model)
        features = pd.read_csv(test_set_location)
        X_train, X_test, y_train, y_test = split_data(features, 0.99)
        X_train, X_test = preprocessing_data(X_train, X_test)
        y_pred_total_beta = model.predict_proba(X_test)
        y_pred_total_beta = reformat(y_pred_total_beta)
        weights = load_weights(weights_location, test_pathologies)
        TestSet_short(y_pred_total_beta, y_test, results_writing='C:\\Users\\David\\PhysioNet_Code\\Independent_Test_Set_One\\Independent_test_set_1\\Results_AB.csv', plot_confusion='C:\\Users\\David\\PhysioNet_Code\\Independent_Test_Set_One\\Independent_test_set_1\\Results_AB_confusion.png')




