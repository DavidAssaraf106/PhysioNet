import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from Global_Experiment_Single_Label import compute_beta_score_CV, split_data, preprocessing_data, \
    TestSet, CrossFoldValidation
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold, f_classif, mutual_info_classif
from scipy.stats import spearmanr, kendalltau
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import csv


warnings.filterwarnings("ignore")

if os.name == 'posix':
    features_locations = '/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Features.csv'
    results_location = '/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Post Filter/'

if os.name == 'nt':
    features_locations = 'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Experiments\\drift\\Experiment 0.5 HZ'
    results_location = 'C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\RFE\\'


def get_final_filtering():
    data = pd.read_csv(features_locations)
    Raw_Label = data['Label']
    le = LabelEncoder()
    Encoded_label = le.fit_transform(Raw_Label)
    data.drop('Label', axis=1, inplace=True)
    data.dropna(inplace=True)
    data['Label'] = Encoded_label
    data_bis = data.copy()
    data_bis.drop('Label', axis=1, inplace=True)
    data_bis.dropna(inplace=True)
    features = data_bis.columns  # features: without label
    variance = 0.01
    anova = 60
    spearsman = 0.025
    info = 0.06
    kendall = 0.1
    sel = VarianceThreshold(threshold=variance)
    sel.fit(data_bis)
    features_variance = features[sel.get_support().tolist()]
    transformer = GenericUnivariateSelect(f_classif, mode='percentile', param=anova)
    transformer.fit(data_bis[features_variance], Encoded_label)
    features_anova_variance = features_variance[transformer.get_support().tolist()].values.tolist()
    features_anova_variance.append('Label')
    spearman_correlation, spearman_p_values = spearmanr(data[features_anova_variance])
    spearman_series = pd.Series(spearman_p_values[-1], index=features_anova_variance).sort_values(ascending=False)
    features_spearman = spearman_series[spearman_series < spearsman].index.values
    features_anova_variance_spearman_kendall = []
    for j, variable in enumerate(features_spearman[:-1]):
        kendall_correlation, kendall_p_value = kendalltau(data_bis[variable], Encoded_label)
        if kendall_p_value < kendall:
            features_anova_variance_spearman_kendall.append(variable)
    score_func_info = np.asarray(mutual_info_classif(data[features_anova_variance_spearman_kendall],
                                                     Encoded_label, discrete_features='auto',
                                                     n_neighbors=5, copy=True,
                                                     random_state=None))
    bool = score_func_info > info
    features_index = np.asarray([i for i in range(len(bool)) if bool[i]])
    features_info = np.asarray(np.asarray(features_anova_variance_spearman_kendall)[features_index])
    features_final = np.append(features_info, 'Label')
    return features_final


def BuildClassifier(features, string):
    X_train, X_test, y_train, y_test = split_data(features)
    X_train, X_test = preprocessing_data(X_train, X_test)
    model_selected = CrossFoldValidation(X_train, y_train)
    dict_results = model_selected.cv_results_
    index = np.argwhere(dict_results['rank_test_score'] == 1)
    index = index[0][0]
    mean_score = dict_results['mean_test_score'][index]
    std_score = dict_results['std_test_score'][index] * 2
    f_beta, g_beta, geometric_mean, f_beta_dict, g_beta_dict = TestSet(X_test, y_test, model_selected)
    features_names = features.columns
    features_number = len(features_names)
    with open(results_location + string, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Results we got with ', features_number, 'features: ', features_names])
        writer.writerow(['Results of the best model on validation set ', mean_score, '+/-', std_score])
        writer.writerow(['Mean F_beta result of the best model on test set ', f_beta])
        writer.writerow(['Mean G_beta result of the best model on test set ', g_beta])
        writer.writerow(['F_beta classes results of the best model on test set ', f_beta_dict])
        writer.writerow(['G_Beta classes results of the best model on test set ', g_beta_dict])
        writer.writerow(['Harmonic Mean results of the best model on test set ', geometric_mean])
    return f_beta


def RFE_method_patch(forward, floating, list=[]):
    cv_scores = []
    std_scores = []
    y_min = []
    y_max = []
    scoring_beta = make_scorer(compute_beta_score_CV, greater_is_better=True)
    data = pd.read_csv(features_locations)
    features_selected_filtering = get_final_filtering()
    data = data[features_selected_filtering]
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.01)
    X_train, X_test = preprocessing_data(X_train, X_test)
    Label = LabelEncoder().fit_transform(data['Label'])
    data.drop('Label', axis=1, inplace=True)
    data['Label'] = Label
    estimator = RandomForestClassifier(n_estimators=150,
                                       max_features='auto',
                                       max_depth=70,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       bootstrap=True,
                                       class_weight='balanced_subsample')
    if list is None and forward is True:
        sfs = SFS(estimator, k_features=(26, 30), forward=forward, floating=floating, verbose=2,
                  scoring=scoring_beta, cv=10, n_jobs=30)
    if list is None and forward is False:
        sfs = SFS(estimator, k_features=(350, 370), forward=forward, floating=floating, verbose=2,
                  scoring=scoring_beta, cv=10, n_jobs=30)
    if list is not None and forward is True:
        sfs = SFS(estimator, k_features=(26, 30), forward=forward, floating=floating, verbose=2,
                  scoring=scoring_beta, cv=10, n_jobs=30, fixed_features=list)
    if list is not None and forward is False:
        sfs = SFS(estimator, k_features=(350, 370), forward=forward, floating=floating, verbose=2,
                  scoring=scoring_beta, cv=10, n_jobs=30, fixed_features=list)
    sfs = sfs.fit(X_train, y_train)
    for i, result in enumerate(sfs.subsets_):
        results = sfs.subsets_.get(result)
        cross_validation_score = results.get('avg_score')
        cv_scores.append(cross_validation_score)
        std_score = np.std(results.get('cv_scores'))
        std_scores.append(std_score)
        y_min.append(cross_validation_score - std_score)
        y_max.append(cross_validation_score + std_score)
    features_selected_final = sfs.k_feature_idx_
    features_selected_list_final = [item for item in features_selected_final]
    with open(results_location + 'forward = ' + str(forward) + ', floating = ' + str(floating) + '.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow([features_selected_list_final, cv_scores, std_scores, y_min, y_max])
    features_classifier = features_selected_filtering[features_selected_final]
    if 'Label' not in features_classifier:
        features_classifier.append('Label')
    return BuildClassifier(data[features_classifier],
                           'forward = ' + str(forward) + ', floating = ' + str(floating) + '.csv')


if __name__ == '__main__':
    RFE_method_patch(forward=True, floating=False, list=[15, 42, 45, 51, 61, 106, 142, 173, 175, 190, 256, 259, 294,
                                                         346, 376, 381, 427, 465])
    RFE_method_patch(forward=False, floating=False)
    RFE_method_patch(forward=False, floating=True)
