import numpy as np
import mifsmaster.mifs as mifs
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold, f_classif, mutual_info_classif
import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import pymrmr
from mRMRmaster import mrmr
from Global_Experiment_Single_Label import BuildClassifier
from FCBF_module import FCBFK

'''
Be careful to where the results are located: check the result_location and plot_location. 
'''
result_location = '/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Post Filter/Multivariate Filters on Selected Features.csv'
plot_location = ''

# Visualisation of overfitting thanks to RF feature importance: test score and validation score + CI
def CVfeatures(model, data, plot_location, plot=False):
    number_features = data.shape[1]  # number of features of the global model
    list = data.columns[:-1]
    feature_imp = pd.Series(model.best_estimator_.feature_importances_, index=list).sort_values(ascending=False)
    list_features = feature_imp.index.values.tolist()
    list_features.insert(0, 'Label')
    mean_validation_scores = []
    std_validation_scores = []
    test_set_score = []
    y_min = []
    y_max = []
    indexes = np.arange(20, number_features, 10)
    for i, index in enumerate(indexes):  # todo: test
        list_n = list_features[:index]
        _, test_score, mean_validation, std_validation = BuildClassifier(data[list_n], write=False,
                                                                         visualize_overfitting=True)
        mean_validation_scores.append(mean_validation)
        std_validation_scores.append(std_validation)
        test_set_score.append(test_score)
        y_min.append(mean_validation - std_validation)
        y_max.append(mean_validation + std_validation)
    if plot:
        plt.figure(figsize=(25, 10))
        for i in range(len(indexes)):
            plt.plot((indexes[i], indexes[i]), (y_min[i], y_max[i]), color='blue', linestyle='dashed', marker='_')
        plt.plot(indexes, mean_validation_scores, color='red', label='Validation score and its CI')
        plt.plot(indexes, test_set_score, color='green', label='Test Score')
        plt.xlabel('Number of features, selected according to their RF importance')
        plt.ylabel('Scores')
        plt.legend()
        plt.savefig(plot_location + 'Overfitting.png')
        plt.cla()
        plt.clf()
        plt.close()
    return mean_validation_scores

# perform mRMR without setting the number of features selected: too harsh on features, not selected.
def feature_selection_MRMR(data, method_mrmr):
    X = data[data.columns[:-1]]
    y = data['Label']
    le = LabelEncoder()
    label = le.fit_transform(y)
    feat_selector = mifs.MutualInformationFeatureSelector(method=method_mrmr, verbose=0)
    feat_selector.fit(X, label)
    return X.columns[feat_selector._get_support_mask()].tolist(), X.columns[feat_selector._get_support_mask()]

#  Implementation based on the pymrmr C code of the authors of the article, not selected either
def feature_selection_mRMR(data, method, number_features):
    data_mRMR = data.copy()
    label = data_mRMR['Label']
    leE = LabelEncoder()
    Label_Encoded = leE.fit_transform(label)
    data_mRMR.drop('Label', axis=1, inplace=True)
    data_mRMR.insert(0, 'Label', Label_Encoded)
    return pymrmr.mRMR(data_mRMR, method, number_features)

def MRMR(data, number_features):
    data_mRMR = data.copy()
    features = data.columns
    label = data_mRMR['Label']
    leE = LabelEncoder()
    Label_Encoded = leE.fit_transform(label)
    data_mRMR.drop('Label', axis=1, inplace=True)
    scores_f_beta=[]
    for i_mrmr, number in enumerate(number_features):
        selection = mrmr.MRMR(n_features=number)
        indexes = selection.fit(data_mRMR.values, Label_Encoded)
        features_selected = features[indexes].values
        if 'Label' not in features_selected:
            features_selected = np.append(features_selected, 'Label')
        model_mrmr, score_mrmr = BuildClassifier(data[features_selected])
        scores_f_beta.append(score_mrmr)
    best_index = np.argmax(scores_f_beta)[0]
    best_features = features_selected[best_index]
    return scores_f_beta, best_features


#  Selected implementation of mRMR, uses FCBF_module: find the best number of features and return the best subset of features
#  Np: the selected one comes from mRMRmaster
def MRMR_FSmethod(data, features_number):
    scores_f_beta = []
    features = []
    for i_mrmr, number in enumerate(features_number):
        features_mrmr = feature_selection_mRMR(data, 'MIQ', number)
        features.append(features_mrmr)
        if not features_mrmr.contains('Label'):
            features_mrmr.append('Label')
        model_mrmr, score_mrmr = BuildClassifier(data[features_mrmr])
        scores_f_beta.append(score_mrmr)
    best_index = np.argmax(scores_f_beta)[0]
    best_features = features[best_index]
    return scores_f_beta, best_features

# Implementation of Correlation Feature Selection method, find the best number of features and return the best subset of features
def CFS(data, features_number):
    data_bis = data.copy()
    Label = data_bis['Label']
    le_here = LabelEncoder()
    Encoded_label = le_here.fit_transform(Label)
    data_bis.drop('Label', axis=1, inplace=True)
    features = data_bis.columns
    scores_f_beta = []
    features_list = []
    for i, number in enumerate(features_number):
        fcbfk = FCBFK(k=number)
        fcbfk.fit(data_bis.values, Encoded_label)
        features_cfs = features[fcbfk.idx_sel].values.tolist()
        features_cfs.append('Label')
        features_list.append(features_cfs)
        model_cfs, score_cfs = BuildClassifier(data[features_cfs])
        scores_f_beta.append(score_cfs)
    # best_index = np.argmax(scores_f_beta)[0]
    # best_features = features_list[best_index]
    # return scores_f_beta, best_features

def variance_test(data, threshold_variance, features=[]):
    data.drop('Label', axis=1, inplace=True)
    if not features:
        features = data.columns
    sel = VarianceThreshold(threshold=threshold_variance)
    sel.fit(data)
    features_variance = features[sel.get_support().tolist()]
    return features_variance

def pearson_test(data, threshold_pearson, features=[]):
    if not features:
        features = data.columns
    correlation = data[features].corr()['Label']
    features_pearson = correlation[correlation > threshold_pearson].index.values
    return features_pearson

def anova_test(data,  thresholds_anova, features=[]):
    label = data['Label'].values
    data.drop('Label', axis=1, inplace=True)
    if not features:
        features = data.columns
    transformer = GenericUnivariateSelect(f_classif, mode='percentile', param=thresholds_anova)
    transformer.fit(data[features], label)
    features_selected = transformer.get_support().tolist()
    features_selected.append(True)
    features_selected_anova = features[features_selected]
    return  features_selected_anova

def MI_test(data, thresholds_info, features=[]):
    label = data['Label'].values
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    data.drop('Label', axis=1, inplace=True)
    if not features:
        features = data.columns
    score_func_info = mutual_info_classif(data, label_encoded, discrete_features='auto', n_neighbors=5, copy=True,
                                          random_state=None)
    features_info = features[score_func_info > thresholds_info].values.tolist()
    features_info.append('Label')
    return features_info

def Spearsman_test(data, thresholds_spearsman, features=[]):
    label = data['Label'].values
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    data.drop('Label', axis=1, inplace=True)
    data['Label'] = label_encoded
    if not features:
        features = data.columns
    spearman_correlation, spearman_p_values = spearmanr(data[features])
    spearman = pd.Series(spearman_p_values[-1], index=features).sort_values(ascending=False)
    scores_spearsman = []
    features_spearman = spearman[spearman < thresholds_spearsman].index.values
    return features_spearman

def Kendall_test(data, thresholds_kendall, features=[]):
    label = data['Label'].values
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    data.drop('Label', axis=1, inplace=True)
    data['Label'] = label_encoded
    if not features:
        features = data.columns
    kendall_features = []
    for j, variable in enumerate(features):
        kendall_correlation, kendall_p_value = kendalltau(data[variable], label_encoded)
        if kendall_p_value < thresholds_kendall:
            kendall_features.append(variable)
    return  kendall_features