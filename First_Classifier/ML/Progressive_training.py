import csv

from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.metrics import plot_confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from Global_Experiment_Single_Label import preprocessing_data, \
    load_challenge_data, split_data, \
    get_freq, VECGAng, extraction_feature_wavedet, extraction_feature_wavedet_BBB
from Scoring_file import compute_challenge_metric, compute_beta_measures, compute_confusion_matrices, \
    compute_modified_confusion_matrix, load_weights
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import random
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
import random

matlab_path1 = "C:\\Users\\David\\ecg-kit-master\\common\\wavedet"
matlab_path2 = "C:\\Users\\David\\ecg-kit-master\\common"
data_directory = 'C:\\Users\\David\\PhysioNet_Code\\Training_complete'
training_location = 'C:\\Users\\David\PhysioNet_Code\\First_Classifier\\Utils\\Experiments\\New_Features\\Experiment_new_features\\Incremental_results\\train_filenames.csv'
training_location_1 = 'C:\\Users\\David\PhysioNet_Code\\First_Classifier\\Utils\\Experiments\\New_Features\\Experiment_new_features\\Incremental_results\\database1.csv'
input_directory = 'C:\\Users\\David\PhysioNet_Code\\Training_complete\\'
pathologies_all = ['AF', 'AFL', 'Brady', 'CRBBB', 'IAVB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR',
                   'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'SNR', 'STach', 'SVPB', 'TAb', 'TInv',
                   'VPB']
effective_pathologies = ['AF', 'AFL', 'Brady', 'IAVB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LPR', 'LQRSV', 'LQT', 'NSIVCB',
                         'PR',
                         'PAC', 'PVC', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'SNR', 'STach', 'TAb', 'TInv']
test_pathologies_effective = ['164889003', '164890007', '426627000', '270492004', '713426002', '445118002',
                              '39732003',
                              '164909002', '164947007', '251146004', '111975006', '698252002', '10370003', '284470004',
                              '427172004',
                              '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000',
                              '164934002', '59931005']
hr_pathologies = ['CRBBB', 'LPR', 'PR', 'SVPB']
q_pathologies = ['Brady', 'PVC']
test_pathologies_all = ['164889003', '164890007', '426627000', '713427006', '270492004', '713426002', '445118002',
                        '39732003',
                        '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007',
                        '111975006',
                        '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000',
                        '63593006',
                        '164934002', '59931005', '17338001']
normal = '426783006'
features_selected = [0, 14, 39, 40, 53, 54, 76, 80, 85, 96, 110, 121, 128, 131, 163, 178, 202, 207, 226, 229, 234, 266,
                     269, 275, 286, 293, 303, 327, 328, 395, 402, 416, 419, 429, 430, 438, 454, 495, 500, 509, 529, 566,
                     584, 624, 629, 636, 692, 725, 778, 784, 789, 792, 812, 823, 836, 858, 863, 872, 874, 888, 900, 916,
                     937, 942, 946, 950, 951, 954, 959, 960, 979, 993, 1007, 1025, 1030, 1034, 1035, 1040, 1066, 1076,
                     1091, 1092, 1118, 1119, 1139, 1146, 1147, 1151, 1155, 1160, 1186, 1202, 1204, 1211, 1216, 1256,
                     1258, 1267, 1268, 1271, 1272, 1273, 1289, 1290, 1291, 1298, 1307, 1325, 1331, 1332, 1333, 1334,
                     1335, 1336, 1337, 1347, 1353, 1354, 1357, 1358, 1359]
names_selected = ['cosEn_0', 'SD1_0', 'AFEv_0', 'OriginCount_0', 'PACEV_0', 'Dqrsmax0', 'Dqrsmean0', 'Dqrsmed0',
                  'Dqrsstd0', 'IRmax0', 'IRmedian0', 'IRmean0',
                  'IRstd0', 'ST_segment_mean_0', 'ST_segment_median_0', 'ST_segment_std_0', 'IrrEv_1',
                  'minRR_1', 'RR_ratio_max_1', 'Fmax_1', 'Fmean1', 'Fmed1', 'Fstd1', 'Sqrsmax1', 'Sqrsmed1',
                  'Sqrsmean1',
                  'Sqrssted1', 'IRmax1', 'IRmedian1', 'IRmean1', 'IRstd1', 'medHR_2', 'AVNN_2', 'SDNN_2',
                  'SEM_2', 'RMSSD_2', 'QRSWidth_median_2',
                  'Amax2', 'Amedian2', 'Amean2', 'Astd2', 'IRmax2', 'IRmedian2', 'IRmean2', 'IRstd2',
                  'amplitudes_2_max_2', 'amplitudes_2_mean_2', 'amplitudes_2_median_2', 'amplitudes_2_std_2',
                  'ST_segment_mean_2', 'ST_segment_median_2', 'ST_segment_std_2', 'IrrEv_3', 'SD2_3', 'DFmax3',
                  'DFmean3', 'DFmed3', 'DFstd3', 'Sqrsmax3', 'Sqrsmean3',
                  'Sqrsmed3', 'Sqrssted3', 'Slope_1_3', 'Slope_2_3', 'amplitudes_0_max_3', 'amplitudes_0_mean_3',
                  'amplitudes_0_median_3', 'amplitudes_0_std_3', 'amplitudes_2_max_3', 'amplitudes_2_mean_3',
                  'amplitudes_2_median_3', 'amplitudes_2_std_3', 'RR_ratio_max_4',
                  'RR_ratio_median_4', 'RR_ratio_mean_4', 'PNN20_5', 'MDPR_5', 'MAPR_5', 'RAPR_5', 'Fmax_5', 'Fmean5',
                  'Fmed5', 'Fstd5', 'Sqrsmax5', 'Sqrsmed5',
                  'Sqrsmean5', 'Sqrssted5', 'IRmax5', 'IRmedian5', 'IRmean5', 'IRstd5', 'Slope_1_5', 'Slope_2_5',
                  'amplitudes_1_max_5', 'amplitudes_1_mean_5', 'amplitudes_1_median_5', 'amplitudes_1_std_5',
                  'Fmax_6', 'Fmean6', 'Fmed6', 'Fstd6', 'DFmax6', 'DFmean6', 'DFmed6', 'DFstd6', 'Amax6', 'Amedian6',
                  'Amean6', 'Astd6', 'amplitudes_1_max_6', 'amplitudes_1_mean_6',
                  'amplitudes_1_median_6', 'amplitudes_1_std_6', 'QRSArea_max_7', 'QRSArea_median_7', 'Sqrsmax7',
                  'Sqrsmean7', 'Sqrsmed7', 'Sqrssted7',
                  'QRSArea_mean_7', 'QRSArea_std_7', 'minRR_8', 'CV_8', 'RR_ratio_median_8', 'RR_ratio_mean_8',
                  'ST_segment_mean_8', 'ST_segment_median_8', 'ST_segment_std_8', 'Fmax_9', 'Fmean9', 'Fmed9', 'Fstd9',
                  'minRR_10', 'SD1_10',
                  'RR_ratio_max_10', 'RR_ratio_std_10', 'Dqrsmax10', 'Dqrsmean10', 'Dqrsmed10',
                  'Dqrsstd10', 'IRmax10', 'IRmedian10', 'IRmean10',
                  'IRstd10', 'Slope_1_10', 'Slope_2_10', 'amplitudes_1_max_10', 'amplitudes_1_mean_10',
                  'amplitudes_1_median_10', 'amplitudes_1_std_10', 'PNN50_11',
                  'MDPR_11', 'MAPR_11', 'RAPR_11', 'QRSWidth_median_11', 'QRSWidth_mean_11', 'QRSWidth_std_11',
                  'QRSArea_max_11', 'QRSArea_std_11', 'Dqrsmax11', 'Dqrsmean11', 'Dqrsmed11',
                  'Dqrsstd11', 'IRmax11', 'IRmedian11', 'IRmean11',
                  'IRstd11', 'amplitudes_1_max_11', 'amplitudes_1_mean_11',
                  'amplitudes_1_median_11', 'amplitudes_1_std_11',
                  'stdPR_0', 'stdP_0', 'stdPamp_0', 'medPRseg_0', 'bsqi_0', 'R_amp_m_1', 'medQT_b_1', 'medQT_fre_1',
                  'bsqi_1', 'stdPR_2', 'bsqi_2', 'medTtype_3', 'stdPamp_3', 'medPRseg_3', 'bsqi_3', 'ratio_4',
                  'R_amp_m_5', 'medPR_5', 'bsqi_5', 'QS_6', 'medPRseg_6', 'bsqi_6', 'stdT_7', 'medPRseg_7', 'bsqi_7',
                  'R_amp_std_8', 'medQT_b_8', 'medPR_8', 'medQT_hod_9', 'medPRseg_9', 'QS_10', 'medQT_b_10', 'medPR_10',
                  'medSTvar2_11', 'medPRseg_11', 'Ratio_1', 'Diff_1', 'Ratio_3', 'Diff_3', 'Ratio_4', 'Ratio_4.1',
                  'Diff_4.1', 'Main_deflection.1', 'Maximum_2.2', 'Main_deflection.2', 'Notch_width_2.4', 'Ratio_1.4',
                  'Diff_1.4', 'Ratio_2.4', 'Diff_2.4', 'Ratio_3.4', 'Diff_3.4', 'Ratio_4.4', 'Ratio_1.5', 'Ratio_4.5',
                  'Diff_4.5', 'VECGAng', 'Age', 'Sex', 'Labels', 'Filename']
thresholding = [0.17000000000000004, 0.25000000000000006, 0.23000000000000004, 0.1, 0.23000000000000004, 0.17000000000000004,
                0.1, 0.15000000000000002, 0.1, 0.4700000000000001, 0.31000000000000005, 0.09000000000000001, 0.22000000000000003,
                0.31000000000000005, 0.15000000000000002, 0.32000000000000006, 0.060000000000000005, 0.060000000000000005,
                0.07, 0.12000000000000001, 0.17000000000000004, 0.32000000000000006, 0.18000000000000005, 0.19]

def multi_label_classifier_RF():
    n_estimators_over = [50, 90, 110, 150, 200, 250]
    max_depth_over = [int(x) for x in np.linspace(10, 70, num=5)]
    random_grid = {'estimator__n_estimators': n_estimators_over,
                   'estimator__max_depth': max_depth_over,
                   }
    clf = OneVsRestClassifier(RandomForestClassifier())
    return clf, random_grid


def XGB_classifier():
    random_grid = {'objective': ['binary:logistic'],
                   'learning_rate': np.linspace(0.0001, 0.2, 5),
                   'max_depth': np.arange(3, 7, 1),
                   'verbosity': [1],
                   'min_child_weight': np.arange(10, 30, 5),
                   'subsample': np.arange(0.6, 0.9, 5),
                   'colsample_bytree': np.arange(0.6, 0.9, 5),
                   'colsample_bylevel': np.arange(0.6, 0.9, 5),
                   'reg_alpha': np.arange(1, 5, 5),
                   'reg_lambda': np.arange(1, 5, 5),
                   'num_parallel_tree': np.arange(70, 120, 10),
                   'n_jobs': [-1],
                   'scale_pos_weight': [1]
                   }
    clf = (xgb.XGBClassifier(n_jobs=-1))
    return clf, random_grid


def RF_classifier():
    # Number of trees in random forest
    n_estimators_over = [200, 250, 300, 350, 400, 450, 500]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 200, num=10)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4, 7, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    weighted = ['balanced_subsample']
    # Create the random grid
    random_grid = {'n_estimators': n_estimators_over,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': weighted

                   }
    # Use the random grid to search for best hyperparameters

    rf = RandomForestClassifier()
    return rf, random_grid


def CrossFoldValidation(X_train, y_train, search=None):
    scoring_competition = make_scorer(compute_challenge_metrics_CV, greater_is_better=True)
    classifier, random_grid = multi_label_classifier_RF()
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
        num_classes = len(test_pathologies_effective)
        y_pred_beta = np.zeros((len(y_pred), num_classes))
        y_val_beta = np.zeros((len(y_pred), num_classes))
        for i in range(len(y_pred)):
            labels_pred = np.zeros(num_classes)
            labels_test = np.zeros(num_classes)
            labels_pred[y_pred[i]] = 1
            labels_test[y_val[i]] = 1
            y_pred_beta[i] = labels_pred
            y_val_beta[i] = labels_test
        final_metrics = compute_challenge_metric(weights, y_val_beta, y_pred_beta, test_pathologies_effective,
                                                 normal)

        return final_metrics
    else:
        return 1


def boxplot(data, list_features):
    for i in range(len(data.columns)):
        if data.columns[i] in list_features:
            data.boxplot(column=data.columns[i], by='Label', grid=False, figsize=(25, 10))
            plt.margins(y=0.05)
            plt.savefig(
                '/home/david/Incremental_study/Debugging_new/boxplot_' + data.columns[
                    i] + '.png')
            plt.cla()
            plt.clf()
            plt.close()


def load_features(train_loc, test_loc):
    print('Loading features ...')
    features_train = pd.read_csv(train_loc)
    list_names = features_train.columns.values
    list_names[-1] = 'Filename'
    features_train.columns = list_names
    features_train = features_train.replace([np.inf, -np.inf], 0)
    features_train.dropna(inplace=True)
    features_train.drop_duplicates(keep='first', inplace=True)
    features_test = pd.read_csv(test_loc)
    features_test = features_test.replace([np.inf, -np.inf], 0)
    features_test.dropna(inplace=True)
    features_test.drop_duplicates(keep='first', inplace=True)
    features_train.drop(features_train.columns[0], axis=1, inplace=True)
    features_test.drop(features_test.columns[0], axis=1, inplace=True)
    features_test.drop(features_test.columns[0], axis=1, inplace=True)
    for df in [features_train, features_test]:
        if 'Label' in df.columns:
            df.drop('Label', axis=1, inplace=True)
    print('Training Base: ', features_train)
    print('Testing Base: ', features_test)
    print('Loading selected features ...')
    features_train = features_train[names_selected]
    list_names_test = features_test.columns.values
    list_names_train = features_train.columns.values
    for i, name in enumerate(list_names_test):
        if name not in list_names_train:
            try:
                lead = float(name[-2:])
                list_names_test[i] = name[:-3] + name[-2:]
            except:
                list_names_test[i] = name[:-2] + name[-1]
    features_test.columns = list_names_test
    features_test = features_test[names_selected]
    print('Training base with selected features: ', features_train)
    print('Testing base with selected features: ', features_test)
    features_train.drop('Filename', axis=1, inplace=True)
    features_test.drop('Filename', axis=1, inplace=True)
    return features_train, features_test
    '''
    new_features_train = pd.read_csv('/home/david/Incremental_study/Data/new_features_train.csv')
    new_features_test = pd.read_csv('/home/david/Incremental_study/Data/new_features_test.csv')
    labels_training = features_train['Labels']
    features_train.drop('Labels', axis=1, inplace=True)
    features_train = pd.merge(features_train, new_features_train, on='Filename')
    features_train['Labels'] = labels_training
    labels_testing = features_test['Labels']
    features_test.drop('Labels', axis=1, inplace=True)
    features_test = pd.merge(features_test, new_features_test, on='Filename')
    features_test['Labels'] = labels_testing
    features_train.drop('Filename', axis=1, inplace=True)
    features_test.drop('Filename', axis=1, inplace=True)
    features_train = features_train.replace([np.inf, -np.inf], 0)
    features_train.dropna(inplace=True)
    features_train.drop_duplicates(keep='first', inplace=True)
    features_test = features_test.replace([np.inf, -np.inf], 0)
    features_test.dropna(inplace=True)
    features_test.drop_duplicates(keep='first', inplace=True)
    return features_train, features_test
    '''


def augment(location):
    augmenting = pd.read_csv(location)
    augmenting = augmenting.replace([np.inf, -np.inf], 0)
    index = []
    for i, patho in enumerate(augmenting['Labels'].values):
        if len(patho) == 2:
            index.append(i)
    augmenting.drop(index, axis=0, inplace=True)
    augmenting.dropna(inplace=True)
    augmenting.drop_duplicates(keep='first', inplace=True)
    augmenting.drop(augmenting.columns[0], axis=1, inplace=True)
    augmenting = augmenting.replace([np.inf, -np.inf], 0)
    augmenting.drop('Filename', axis=1, inplace=True)
    augmenting.drop('Label', axis=1, inplace=True)
    encoded_labels = []
    aug_pathologies = []
    for i, patho in enumerate(augmenting['Labels'].values):
        patho = patho.split(',')
        patho[0] = patho[0][1:]
        patho[-1] = patho[-1][:-1]
        patho = [u.strip() for u in patho]
        for path in patho:
            aug_pathologies.append(path)
        sample_i = np.zeros(24)
        for p in patho:
            ind = np.argwhere(np.asarray(effective_pathologies) == p[1:-1].strip())
            sample_i[ind] = 1
        encoded_labels.append(sample_i)
    augmenting['Labels'] = encoded_labels
    return augmenting


def uniformize_columns(augmenting, features_train, features_test):
    columns_aug = augmenting.columns.values
    columns_train = features_train.columns.values
    for i, name in enumerate(columns_aug[:-1]):
        if name not in names_selected:
            try:
                float(name[-2:])
                columns_aug[i] = name[:-3] + name[-2:]
            except:
                columns_aug[i] = name[:-2] + name[-1:]
    augmenting.columns = columns_aug
    return augmenting[names_selected[:-1]]


def get_labels_ovr(augmenting, features_train, features_test):
    y_train = features_train['Labels']
    y_augm = augmenting['Labels']
    y_test = features_test['Labels']
    y_train_new = []
    y_test_new = []
    for i, string in enumerate(y_train):
        labels = string.split('.')
        labels[0] = labels[0][1]
        labels = labels[:-1]
        labels_int = [int(label) for label in labels]
        y_train_new.append(labels_int)
    for i, string in enumerate(y_test):
        labels = string.split('.')
        labels[0] = labels[0][1]
        labels[-1] = labels[0][0]
        labels_int = [int(label) for label in labels]
        y_test_new.append(labels_int)
    for aug in y_augm:
        y_train_new.append(aug.tolist())
    indexes_train = [[value[0] for value in np.argwhere(np.array(label) == 1)] for label in y_train_new]
    indexes_test = [[value[0] for value in np.argwhere(np.array(label) == 1)] for label in y_test_new]
    encoder = MultiLabelBinarizer()
    encoder.fit(indexes_train)
    y_train = encoder.transform(indexes_train)
    y_test = encoder.transform(indexes_test)
    return y_train, y_test


def standardize_m_M(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def standardize(X_train, X_test, X_train_snr):
    list_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    categorical = ['AFEv_', 'OriginCount_', 'IrrEv_', 'PACEV_']
    categorical_features = [cat + str(lead) for cat in categorical for lead in list_lead]
    categorical_features.append('Sex')
    numerical_features = [name for name in X_train.columns.values if name not in categorical_features]
    t = [('num', StandardScaler(), numerical_features)]
    col_transform = ColumnTransformer(transformers=t)
    col_transform.fit(X_train)
    imputer = SimpleImputer().fit(X_train)
    X_train[numerical_features] = col_transform.transform(X_train)
    X_test[numerical_features] = col_transform.transform(X_test)
    X_train_snr[numerical_features] = col_transform.transform(X_train_snr)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    X_test = imputer.transform(X_test)
    joblib.dump(imputer, '/home/david/imputer.sav')
    joblib.dump(col_transform, '/home/david/transformer.sav')
    return X_train, X_test, X_train_snr


def thresholding(y_pred_total_beta, y_test):
    print('We are finding the optimal thresholds for probabilities')
    weights = load_weights('/home/david/weights.csv', test_pathologies_effective)
    num_classes = len(effective_pathologies)  # should be 24
    preds = np.arange(0.05, 0.5, 0.01)
    running_max = 0
    y_thresh_max = []
    u_opt = []
    for p in range(20000):
        u = random.choices(population=preds, k=24)
        y_thresh = []
        for i in range(len(y_test)):
            classes = np.zeros(num_classes)
            for j in range(num_classes):
                classes[j] = y_pred_total_beta[i][j] > u[j]
            y_pred_i = np.zeros(num_classes)
            if len(np.argwhere(classes)) > 0:
                for k in range(len(classes)):
                    if classes[k]:
                        y_pred_i[k] = 1
            else:
                ind = np.argmax(y_pred_total_beta[i])
                y_pred_i[ind] = 1
            y_thresh.append(y_pred_i.tolist())
        y_thresh = np.array(y_thresh)
        final_metrics = compute_challenge_metric(weights, y_test, y_thresh, test_pathologies_effective,
                                                 normal)
        if final_metrics > running_max:
            running_max = final_metrics
            print('Running max: ', running_max)
            y_thresh_max = y_thresh
            u_opt = u
    print('The optimal thresholding is', u_opt)
    print('The optimal thresholding score is', running_max)
    return y_thresh_max


def plot_confusion_matrix_all(num_classes, B, plot_confusion):
    f, axarr = plt.subplots(1, 2, figsize=(25, 20))
    axarr[0] = plt.subplot(211)
    axarr[0].matshow(B, cmap=plt.cm.Blues)
    for i in range(num_classes):
        for j in range(num_classes):
            c = int(B[j, i])
            axarr[0].text(i, j, str(c), va='center', ha='center')
    axarr[0].set_xticks(np.arange(len(effective_pathologies)))
    axarr[0].set_yticks(np.arange(len(effective_pathologies)))
    axarr[0].set_xticklabels(effective_pathologies, rotation='vertical')
    axarr[0].set_yticklabels(effective_pathologies)
    axarr[0].set_xlabel('Predicted Label')
    axarr[1] = plt.subplot(212)
    axarr[1].matshow(weights, cmap=plt.cm.Blues)
    for i in range(num_classes):
        for j in range(num_classes):
            c = weights[j, i]
            axarr[1].text(i, j, str(c), va='center', ha='center')
    f.savefig(plot_confusion)


def Testing_model(y_test, y_thresh, model_selected, result_writing, plot_confusion):
    print('Testing the model ..')
    num_classes = len(effective_pathologies)
    weights = load_weights('/home/david/weights.csv', test_pathologies_effective)
    A = compute_confusion_matrices(y_test, y_thresh)
    fbeta, gbeta, fbeta_measure, gbeta_measure = compute_beta_measures(y_test, y_thresh, 2)
    fbeta_dict, gbeta_dict = dict(zip(effective_pathologies, fbeta_measure)), dict(
        zip(effective_pathologies, gbeta_measure))
    final_metrics = compute_challenge_metric(weights, y_test, y_thresh, test_pathologies_effective,
                                             normal)
    with open(result_writing, 'a+', newline='') as f:
        writer = csv.writer(f)
        for i, matrix in enumerate(A):
            writer.writerow(['Confusion matrix for the pathology', effective_pathologies[i]])
            writer.writerow([matrix])
    B = compute_modified_confusion_matrix(y_test, y_thresh)
    print('Plotting the confusion matrix')
    plot_confusion_matrix_all(num_classes, B, plot_confusion)
    with open(result_writing, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Mean F_beta result of the best model on test set ', fbeta])
        writer.writerow(['Mean G_beta result of the best model on test set ', gbeta])
        writer.writerow(['F_beta classes results of the best model on test set ', fbeta_dict])
        writer.writerow(['G_Beta classes results of the best model on test set ', gbeta_dict])
        writer.writerow(['Competition results of the best model on test set ', final_metrics])
    try:
        dict_results = model_selected.cv_results_
        index = np.argwhere(dict_results['rank_test_score'] == 1)[0][0]
        mean_score = dict_results['mean_test_score'][index]
        std_score = dict_results['std_test_score'][index] * 2
        writer.writerow(['The best parameters are', model_selected.best_params_])
        writer.writerow(['Results of the best model on validation set ', mean_score, '+/-', std_score])
    except:
        return


def load_ovr_model():
    model_selected = joblib.load('/home/david/model_ovr.sav')
    return model_selected


def load_snr_model():
    classifier = joblib.load('/home/david/model_snr.sav')
    return classifier


def get_labels_SNR_patho(features_train, y_train):
    features_train.reset_index(inplace=True)
    indexes_snr = []
    for i, label in enumerate(y_train):
        if label[20] == 1:
            indexes_snr.append(1)
            y_train[i][20] = 0
        else:
            indexes_snr.append(0)
    features_train_snr = features_train[features_train.columns[:-1]]
    y_train_snr = indexes_snr
    return features_train_snr, features_train, y_train_snr, y_train


def harmonize_predictions(y_pred_total_beta_snr, y_pred_total_beta_patho):
    y_pred_total_beta = []
    for i, pred in enumerate(y_pred_total_beta_patho):
        pred[20] = y_pred_total_beta_snr[i][1]
        y_pred_total_beta.append(pred)
    return y_pred_total_beta


def y_snr(y_test):
    y_snr = []
    for label in y_test:
        if label[20] == 1:
            y_snr.append(1)
        else:
            y_snr.append(0)
    return y_snr


def snr_model(X_train_snr, y_train_snr, X_test, y_test_snr):
    scoring_competition = make_scorer(compute_challenge_metrics_CV, greater_is_better=True)
    print('Working on the SNR Vs NON SNR model ...')
    print(np.mean(y_test_snr))
    classifier, random_grid = RF_classifier()
    random_model = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, verbose=1,
                                      scoring=scoring_competition,
                                      n_jobs=-1, return_train_score=True, cv=5, refit=True)
    random_model.fit(X_train_snr, y_train_snr)
    plot_confusion_matrix(random_model, X_test, y_test_snr)
    plt.savefig('/home/david/Incremental_study/Results/SNR_model_scoring_GS.png')
    dict_results = random_model.cv_results_
    print(dict_results)
    index = np.argwhere(dict_results['rank_test_score'] == 1)[0][0]
    mean_score = dict_results['mean_test_score'][index]
    std_score = dict_results['std_test_score'][index] * 2
    print('The best parameters are', random_model.best_params_)
    print(mean_score)
    print(std_score)
    predictions = np.array(random_model.predict_proba(X_test))
    predictions = predictions[:, 1]
    thresholds = np.arange(0.1, 0.8, 0.01)
    running_score = 0
    running_thresh = 0
    for t in thresholds:
        y_thresh = predictions > t
        acc = np.mean(y_thresh == y_test_snr)
        if acc > running_score:
            running_score = acc
            running_thresh = t
            print(running_score)
            print(running_thresh)


def classifier_chain(X_train_patho, y_train_patho, X_test):
    chains = [ClassifierChain(RandomForestClassifier(), order='random', random_state=i)
              for i in range(10)]
    for chain in chains:
        chain.fit(X_train_patho, y_train_patho)
    Y_pred_chains = np.array([chain.predict_proba(X_test) for chain in
                              chains])
    Y_pred_ensemble = Y_pred_chains.mean(axis=0)
    return Y_pred_ensemble, chains


def training_testing(result_writing, plot_confusion):
    """
    Non processed database for now. The augmenting features are processed.
    """
    testing_location = '/home/david/Incremental_study/Data/features_test_multilabelled.csv'
    database = '/home/david/Incremental_study/Data/features_train_multilabel_final.csv'
    location_aug = '/home/david/Incremental_study/Data/Preprocessed_data/features_test_prep.csv'
    features_train, features_test = load_features(database, testing_location)
    augmenting = augment(location_aug)
    augmenting = uniformize_columns(augmenting, features_train, features_test)
    y_train, y_test = get_labels_ovr(augmenting, features_train, features_test)
    features_train = pd.concat([features_train, augmenting], axis=0)
    features_train_snr, features_train_patho, y_train_snr, y_train_patho = get_labels_SNR_patho(features_train, y_train)
    print('Label composition of the training set: ', dict(zip(effective_pathologies, np.sum(y_train_patho, axis=0))))
    print('Label composition of the testing set: ', dict(zip(effective_pathologies, np.sum(y_test, axis=0))))
    X_train_patho = features_train[features_train.columns[1:-1]]
    X_train_snr = features_train_snr[features_train_snr.columns[1:]]
    X_test = features_test[features_test.columns[:-1]]
    X_train_patho, X_test, X_train_snr = standardize(X_train_patho, X_test, X_train_snr)
    print('Training the SNR vs nonSNR model... ')
    print('Training features ...')
    print(X_train_snr.shape)
    print('Training labels ...')
    print(len(y_train_snr))
    y_test_snr = y_snr(y_test)
    model_snr = load_snr_model()
    # model_snr.fit(X_train_snr, y_train_snr)
    plot_confusion_matrix(model_snr, X_test, y_test_snr)
    plt.savefig('/home/david/Incremental_study/Results/SNR_model_XGB.png')
    model_ovr = load_ovr_model()
    print('Training the One Vs Rest model for multi-label classification... ')
    print('Training features ...')
    print(X_train_patho.shape)
    print('Training labels ...')
    print(len(y_train_patho))
    print(np.sum(y_train_patho, axis=0))
    #model_ovr.fit(X_train_patho, y_train_patho)
    #joblib.dump(model_ovr, '/home/david/model_ovr.sav')
    #joblib.dump(model_snr, '/home/david/model_snr.sav')
    X_test = np.nan_to_num(X_test)
    y_pred_total_beta_patho = model_ovr.predict_proba(X_test)
    # y_pred_total_beta_patho, model_ovr = classifier_chain(X_train_patho, y_train_patho, X_test)
    y_pred_total_beta_snr = model_snr.predict_proba(X_test)
    y_pred_total_beta = harmonize_predictions(y_pred_total_beta_snr, y_pred_total_beta_patho)
    pd.DataFrame(y_pred_total_beta).to_csv('/home/david/Incremental_study/Results/Multi_label/preds_RF.csv')
    y_thresh = np.array(thresholding(y_pred_total_beta, y_test))
    Testing_model(y_test, y_thresh, model_ovr, result_writing, plot_confusion)
    return model_ovr


def BuildClassifier(features, result_writing, write=True, search=None, results_plot_location=None):
    X_train, X_test, y_train, y_test = split_data(features)
    X_train, X_test = preprocessing_data(X_train, X_test)
    model_selected = CrossFoldValidation(X_train, y_train, search=None)
    dict_results = model_selected.cv_results_
    print(dict_results)
    y_pred = model_selected.predict_proba(X_test)
    print(y_pred)
    indices_selected = np.squeeze([np.argmax(prediction) for prediction in y_pred])
    print(pd.Series(indices_selected).value_counts())
    print(pd.Series(y_test).value_counts())


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data

# todo:XGBoost resampling
# todo: work on bsqi
if __name__ == '__main__':
    experiment_1 = 'done'
    experiment_filename = 'done'
    experiment_joachim = ''
    results_experiments = {"done", "wait", "fail"}
    if experiment_1 not in results_experiments:
        weights = load_weights('/home/david/weights.csv', test_pathologies_effective)
        training_testing('/home/david/Incremental_study/Results/Results_different_stand.csv',
                         '/home/david/Incremental_study/Results/Confusion_different_stand.png')
    if experiment_filename not in results_experiments:
        training = pd.read_csv('/home/david/Incremental_study/Data/features_train_multilabel.csv')
        filenames = pd.read_csv('/home/david/Incremental_study/Data/features_train_multilabelled.csv')['Filename']
        print(len(training))
        print(len(filenames))
        print(training.isna().sum())
        training_location = pd.read_csv
        ref = pd.read_csv('/home/david/Incremental_study/Ref_split/train_filenames.csv')
        news = ref['FileName'].tail(1599)  # ok: only SNR
        adds = pd.concat([filenames, news], axis=0)
        training = pd.concat([training, adds], axis=1)
        training.drop(training.columns[0], axis=1, inplace=True)
        print(training)
        training.dropna(inplace=True)
        print(training)
        training.to_csv('/home/david/Incremental_study/Data/features_train_multilabel_final.csv')
    if experiment_joachim not in results_experiments:
        considered = set()
        for i, f in enumerate(os.listdir(input_directory)):
            if f.endswith('mat') and f[0] not in considered:
                tmp_input_file = os.path.join(input_directory, f)
                data, header_data = load_challenge_data(tmp_input_file)
                print(header_data)
                considered.add(f[0])
                if len(considered) == 6:
                    break
