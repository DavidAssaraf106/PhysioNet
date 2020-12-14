import warnings
import numpy as np
from sklearn.metrics import make_scorer, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold, f_classif, mutual_info_classif
import csv
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# from Global_Experiment_Single_Label import get_features_from_QRS
# from Filtering_Method import CFS, MRMR
# from Wrapper_Methods import RFE_method_patch

warnings.filterwarnings("ignore")

preprocess_method = 'notch'
feature_selection_method = 'MRMR'
features_locations = '/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Features.csv'
results_location = '/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Post Filter/Multivariate Filters on Selected Features.csv'
plot_location = '/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/'
pathologies = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
parameters = {'Notch': 0, 'Drift': 0.5, 'Baseline Wandering': 0, 'Wavelet': 0, 'Filter_bandpass': [0, 0], 'Padding': 0,
              'Mirroring': 0, 'Feature_selection': 'MRMR'}
features_AF = ['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEV', 'AVNN', 'SDNN', 'SEM',
               'minRR', 'medHR', 'PNN20', 'PNN50', 'RMSSD', 'CV', 'SD1', 'SD2']
features_AVB = ['MDPR', 'MAPR', 'RAPR']
features_PAC = ['RR_ratio_max', 'RR_ratio_median', 'RR_ratio_mean', 'RR_ratio_std', 'QRSWidth_max', 'QRSWidth_median',
                'QRSWidth_mean', 'QRSWidth_std', 'QRSArea_max', 'QRSArea_median', 'QRSArea_mean', 'QRSArea_std']
features_PVC = ['Fmax_', 'Fmean', 'Fmed', 'Fstd', 'DFmax', 'DFmean', 'DFmed', 'DFstd', 'Dqrsmax', 'Dqrsmean', 'Dqrsmed',
                'Dqrsstd', 'Sqrsmax', 'Sqrsmed', 'Sqrsmean', 'Sqrssted', 'Amax', 'Amedian', 'Amean', 'Astd', 'IRmax',
                'IRmedian', 'IRmean', 'IRstd']
features_ST = ['amplitudes_0_max', 'amplitudes_0_mean', 'amplitudes_0_median', 'amplitudes_0_std', 'amplitudes_1_max',
               'amplitudes_1_mean', 'amplitudes_1_median', 'amplitudes_1_std', 'amplitudes_2_max', 'amplitudes_2_mean',
               'amplitudes_2_median', 'amplitudes_2_std', 'ST_segment_mean', 'ST_segment_median', 'ST_segment_std']

repartition_pathologies = [features_AF, features_AVB, features_PAC, features_PVC, features_ST]


def split_data(features, test_size=0.2):
    label = features['Label']
    features = features.drop(['Label'], axis=1)
    le = LabelEncoder()
    label = le.fit_transform(label)
    return train_test_split(features, label, test_size=test_size, stratify=label)


def preprocessing_data(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def CrossFoldValidation(X_train, y_train, search=None):
    scoring_beta = make_scorer(compute_beta_score_CV, greater_is_better=True)
    classifier, random_grid = RF_classifier()
    if search is not None:
        random_model = GridSearchCV(estimator=classifier, param_grid=random_grid, verbose=1, scoring=scoring_beta,
                                    n_jobs=20, return_train_score=True, cv=10, refit=True)
    else:
        random_model = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, verbose=1,
                                          n_jobs=20, scoring=scoring_beta, return_train_score=True, cv=10)
    random_model.fit(X_train, y_train)
    return random_model


def compute_beta_score(labels, output, beta, num_classes, plot=False):
    num_recordings = len(labels)
    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l = np.ones(num_classes)

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):

            num_labels = np.sum(labels[i])

            if labels[i][j] and output[i][j]:
                tp += 1 / num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1 / num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1 / num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1 / num_labels

        # Summarize contingency table.
        if ((1 + beta ** 2) * tp + (fn * beta ** 2) + fp):
            fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0
    for i in range(num_classes):
        f_beta += fbeta_l[i] * C_l[i]
        g_beta += gbeta_l[i] * C_l[i]
        f_measure += fmeasure_l[i] * C_l[i]
        accuracy += accuracy_l[i] * C_l[i]

    f_beta_dict = dict(zip(pathologies, fbeta_l))
    g_beta_dict = dict(zip(pathologies, gbeta_l))
    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return accuracy, f_measure, f_beta, g_beta, f_beta_dict, g_beta_dict


def compute_beta_score_CV(y_val, y_pred):
    if (len(y_pred)) > 0:
        num_classes = 9
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
        _, _, f_beta, g_beta, _, _ = (compute_beta_score(y_val_beta, y_pred_beta, 2, num_classes=9))
        # score = np.sqrt(f_beta * g_beta)
        return f_beta
    else:
        return 1


def TestSet(X_test, y_test, model):  # in order to allow multi label classification? no
    num_classes = 9
    y_pred_total_beta = model.predict_proba(X_test)  # size : (len(y_test), num_classes)
    for i in range(len(y_test)):
        classes = y_pred_total_beta[i] > 0.3  # todo: adapt the threshold
        y_pred_i = np.zeros(num_classes)
        if (len(np.argwhere(classes)) > 0):
            for k in range(len(classes)):
                if classes[k]:
                    y_pred_i[k] = 1
                    break  # for the confusion matrix, todo: change later for multi-label classification
        else:
            ind = np.argmax(y_pred_total_beta[i])
            y_pred_i[ind] = 1
        y_pred_total_beta[i] = y_pred_i

    y_test_beta = np.zeros((len(y_test), num_classes))
    for i in range(len(y_test)):
        labels_test = np.zeros(num_classes)
        labels_test[y_test[i]] = 1
        y_test_beta[i] = labels_test

    accuracy, _, f_beta, g_beta, f_beta_dict, g_beta_dict = (
        compute_beta_score(y_test_beta, y_pred_total_beta, 2, num_classes=9,
                           plot=True))  # accuracy, f_measure, f_beta, g_beta

    geometric_mean = np.sqrt(f_beta * g_beta)

    return f_beta, g_beta, geometric_mean, f_beta_dict, g_beta_dict


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


def BuildClassifier(features, result_writing, write=True, visualize_overfitting=False, search=None):
    X_train, X_test, y_train, y_test = split_data(features)
    X_train, X_test = preprocessing_data(X_train, X_test)
    model_selected = CrossFoldValidation(X_train, y_train, search)
    dict_results = model_selected.cv_results_
    index = np.argwhere(dict_results['rank_test_score'] == 1)
    index = index[0][0]
    mean_score = dict_results['mean_test_score'][index]
    std_score = dict_results['std_test_score'][index] * 2
    f_beta, g_beta, geometric_mean, f_beta_dict, g_beta_dict = TestSet(X_test, y_test, model_selected)
    features_names = features.columns
    features_number = len(features_names)
    if write:
        with open(result_writing, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['The parameters tuned here are', model_selected.best_estimator_])
            writer.writerow(['Results we got with ', features_number, 'features: ', features_names])
            writer.writerow(['Results of the best model on validation set ', mean_score, '+/-', std_score])
            writer.writerow(['Mean F_beta result of the best model on test set ', f_beta])
            writer.writerow(['Mean G_beta result of the best model on test set ', g_beta])
            writer.writerow(['F_beta classes results of the best model on test set ', f_beta_dict])
            writer.writerow(['G_Beta classes results of the best model on test set ', g_beta_dict])
            writer.writerow(['Harmonic Mean results of the best model on test set ', geometric_mean])
    if visualize_overfitting:
        return model_selected, f_beta, mean_score, std_score  # model, test_score, mean validation score, standard deviation validation score
    plot_confusion_matrix(model_selected.best_estimator_, X_test, y_test)
    return model_selected, f_beta


def get_final_filtering(index=False):
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
    if not index:
        return features_final
    else:
        return features_index


def RFE_method_patch(forward, floating, data_location, results_location):
    cv_scores = []
    std_scores = []
    y_min = []
    y_max = []
    scoring_beta = make_scorer(compute_beta_score_CV, greater_is_better=True)
    data = pd.read_csv(data_location)
    data = data.replace([np.inf, -np.inf], 0).dropna(axis=0)
    data = data.replace([float('inf'), -float('inf')], 0).dropna(axis=0)
    data.dropna(inplace=True)
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

    sfs = SFS(estimator, k_features=(50, 150), forward=forward, floating=floating, verbose=2,
              scoring=scoring_beta, cv=6, n_jobs=40)
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
    with open(results_location + 'forward = ' + str(forward) + ', floating = ' + str(
            floating) + '_new_feature' + '.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Eighth Experiment: Analyzing the performances of Feature extraction on the challenge features extracted.'])
        writer.writerow([features_selected_list_final, cv_scores, std_scores, y_min, y_max])



def repartition(list):
    list_count = [[], [], [], [], []]
    for j, features in enumerate(list):
        if features[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            features = features[:-1]
        if features[-1] == '1':
            features = features[:-1]
        if features[-1] == '_':
            features = features[:-1]
        if features[-1] == '.':
            features = features[:-1]
        for i, list_pathology in enumerate(repartition_pathologies):
            if features in list_pathology:
                list_count[i].append(features)
    list_count_number = [len(list_count[i]) for i in range(len(list_count))]
    return list_count, list_count_number


if __name__ == '__main__':
    experiment_1 = 'done'
    experiment_2 = 'done'
    experiment_3 = 'done'
    experiment_4 = 'fail'   #todo: do this experience again, the results were bad
    experiment_5 = 'done'
    experiment_6 = 'fail'  # todo: refais, ça beug.
    experiment_7 = 'done'
    experiment_8 = ''
    experiment_9 = ''
    results_experiments = {'done', 'fail'}
    '''
    First experience: test the performances of the notch filter, between 48 and 52Hz.
    Issue: the csv file has 7379 lines, meaning that there should be some duplicates in rows.
    Feature location on server : "/home/david/Utils/Experiments/notch/Experiment_number_4/Features.csv"
    Results location on server: "/home/david/Utils/Experiments/notch/Experiment_number_4/results.csv"
    '''
    if experiment_1 not in results_experiments:
        print('Performing Experiment 1: test the performances of the notch filter')
        features_notch = pd.read_csv("/home/david/Utils/Experiments/notch/Experiment_number_4/Features.csv")
        results_notch = "/home/david/Utils/Experiments/notch/Experiment_number_4/results.csv"
        filtered_features = get_final_filtering()
        with open(results_notch, 'w', newline='') as file_notch:
            writer = csv.writer(file_notch)
            writer.writerow(['First experience: test the performances of the notch filter, between 48 and 52Hz.'])
            writer.writerow(
                ['We extracted sample from ', len(results_notch), 'Patients, and Here are the results we got'])
            writer.writerow(['First, Grid Search CV without selecting the filtered features'])
        BuildClassifier(features_notch, results_notch, search='gs')
        with open(results_notch, 'a+', newline='') as file_notch:
            writer = csv.writer(file_notch)
            writer.writerow(['Second, with selecting the filtered features'])
        BuildClassifier(features_notch[filtered_features], results_notch, search='gs')
    '''
    Second experiment: test the performances of the final preprocess filter we designed.
    The same issue is encountered
    Features location on server : "/home/david/Utils/Experiments/Final/Experiment_number_5/Features.csv"
    Results location on server : "/home/david/Utils/Experiments/Final/Experiment_number_5/Results.csv"
    '''
    if experiment_2 not in results_experiments:
        print('Experiment 2: Testing the final filter we designed')
        features_final = pd.read_csv("/home/david/Utils/Experiments/Final/Experiment_number_5/Features.csv")
        results_final = "/home/david/Utils/Experiments/Final/Experiment_number_5/Results.csv"
        features_final.dropna(inplace=True)
        features_final.drop_duplicates(keep='first', inplace=True)
        features_final = features_final.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        features_final = features_final.replace([float('inf'), -float('inf')], np.nan).dropna(axis=0)
        with open(results_final, 'w', newline='') as file_final:
            writer_final = csv.writer(file_final)
            writer_final.writerow(
                ['Second experiment: test the performances of the final preprocess filter we designed.'])
            writer_final.writerow(
                ['We extracted sample from ', len(results_final), 'Patients, and Here are the results we got'])
            writer_final.writerow(['First, Grid Search CV without selecting the filtered features'])
        BuildClassifier(features_final, results_final, search='gs')
        with open(results_final, 'a+', newline='') as file_final:
            writer_final = csv.writer(file_final)
            writer_final.writerow(['Second, with selecting the filtered features'])
        filtered_features = get_final_filtering(index=True)
        features = features_final.columns[filtered_features]
        features = np.append(features.values, 'Label')
        BuildClassifier(features_final[features], results_final, search='gs')
    '''
    Third Experiment: Analyzing the features selected by the floating forward approach done on the filtered features. 
    '''
    if experiment_3 not in results_experiments:
        print('Experiment 3: Analysis of the features selected by the FF RFE approach')
        features_final = pd.read_csv("/home/david/Utils/Experiments/Final/Experiment_number_5/Features.csv")
        results_final = "/home/david/Utils/Experiments/Final/Experiment_number_5/Results.csv"
        features_final.dropna(inplace=True)
        features_final.drop_duplicates(keep='first', inplace=True)
        features_final = features_final.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        features_final = features_final.replace([float('inf'), -float('inf')], np.nan).dropna(axis=0)
        indexes_selected = [8, 15, 16, 40, 46, 49, 51, 60, 61, 62, 72, 74, 104, 109, 111, 119, 122, 123, 124, 131, 139,
                            141, 142, 173, 176, 181, 182, 190, 201, 207, 238, 259, 264, 273, 298, 304, 315, 319, 348,
                            351, 363, 365, 374, 379, 384, 395, 398, 406, 409, 413, 418, 450, 458, 467, 473, 475, 479, 484]
        names_selected = features_final.columns[indexes_selected]
        _, repartition_features = np.asarray(repartition(names_selected.values))
        features_experiment_floating = "/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Features.csv"
        results_floating = "/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Post Filter/Forward_floating_results.csv"
        data = pd.read_csv(features_experiment_floating)
        data.dropna(inplace=True)
        filtered_features = get_final_filtering(index=False)
        features_selected = filtered_features[indexes_selected]
        features_selected = np.append(features_selected, 'Label')
        with open(results_floating, 'w', newline='') as file_floating:
            writer_floating = csv.writer(file_floating)
            writer_floating.writerow(
                ['Third Experiment: Analyzing the features selected by the floating forward approach done'
                 ' on the filtered features.'])
            writer_floating.writerow(['Indexes of features selected:', indexes_selected])
            writer_floating.writerow(['Repartition of the features selected:', repartition_features])
            writer_floating.writerow(['Performances of this model with Grid Search CV:'])
        BuildClassifier(data[features_selected], results_floating, search='gs')
    '''
    Fourth experiment: Visualization of the performances of Floating forward Method, in order to be inserted in the 
    report
    '''
    if experiment_4 not in results_experiments:
        print('Performing experiment 4: vizualisation of features selected via Forward Floating approach')
        cross_validation_scores = [0.2605880881239258, 0.38618817350195656, 0.47516646860152206, 0.54457680544915,
                                   0.584641016256772, 0.6269092700930124,
                                   0.6517334250520718, 0.6664472269497618, 0.6794475642579999, 0.6879273423396581,
                                   0.6954378241031396, 0.703816222588879,
                                   0.7105114477999326, 0.7112334026077232, 0.7143998470553883, 0.7188303797682546,
                                   0.7212846205516111, 0.7222774642450531,
                                   0.722477452761393, 0.7278279957782066, 0.730176811195998, 0.7302927228728077,
                                   0.7324483344754121, 0.735946455043221,
                                   0.7341019770603862, 0.7358582032459523, 0.7372748066111142, 0.7401786248509898,
                                   0.7413357316491861, 0.7411178606160584,
                                   0.7427838809281662, 0.7440129620929467, 0.7417250576627161, 0.7415822997697661,
                                   0.7412468938880519, 0.7444931438878157,
                                   0.7455007341586949, 0.7475769119212243, 0.7493698267927265, 0.7478214963001801,
                                   0.7476535598048549, 0.7452158738358223,
                                   0.7466226136102662, 0.7461510384634461, 0.7455846017966051, 0.7451708765515445,
                                   0.7466004016139246, 0.7431497773967927,
                                   0.7434832483538779, 0.7488982251514892, 0.7473263436827157, 0.7476183626384257,
                                   0.7490976105339975, 0.7495987847726786,
                                   0.7487818594958505, 0.7503178199079236, 0.7487971886454607, 0.7532361462359498,
                                   0.7513196559100241, 0.7504789530805912,
                                   0.748788216790008, 0.7475938368791839, 0.7496201478229659, 0.7505372714458763,
                                   0.7501522401316767, 0.7492028206418585,
                                   0.750663311754589, 0.7495575298906516, 0.7506260878233322, 0.74887544961518,
                                   0.7493869960578481, 0.7500506368803596,
                                   0.7501316105656425, 0.7485643696449982, 0.7492889022618874, 0.7487360046792212,
                                   0.7497624952775667, 0.7483413857814476,
                                   0.7491088327704477, 0.7503986854598108, 0.749561894148742, 0.7500760047720902,
                                   0.7498806780915684, 0.7496657102162843,
                                   0.7478957753962191, 0.74987002433905, 0.7482063751345315, 0.7477604582020358,
                                   0.7491336292512233, 0.7480465021695709,
                                   0.7480766087651078, 0.7468028461411634, 0.7466696203988844, 0.7482709858384079,
                                   0.745707873612819, 0.7464248115916523,
                                   0.7459359521365627, 0.7451616768939654, 0.7447073795702562, 0.7463802158170271]
        standard_deviation = [0.012523124426808833, 0.004040476446754646, 0.013753873080876066, 0.01662823693965504,
                              0.01512294306836487,
                              0.013166405134717956, 0.016962956480547877, 0.01461450161669251, 0.008672379921293868,
                              0.008278583954772493,
                              0.014162860517803879, 0.013178307923928984, 0.012835418495047057, 0.00960589661317563,
                              0.012718471508359232,
                              0.008337097046599554, 0.01188360277439471, 0.009736400706559983, 0.013612493527336303,
                              0.008608799994107446,
                              0.011965093691188568, 0.010221940980042112, 0.012702337469054095, 0.019427836742485948,
                              0.014430414575818673,
                              0.012495416148899685, 0.01601751704874803, 0.010167251599964714, 0.013297227528637292,
                              0.007476990021485804,
                              0.011483572519772386, 0.01391039385200184, 0.0077008356459023425, 0.01103703800708456,
                              0.0070339081107677615,
                              0.011118780143293669, 0.013219453259212558, 0.011598772163175801, 0.014848915176134032,
                              0.008178278598825408,
                              0.008057045749464161, 0.008476916813611927, 0.008380716241103576, 0.008288355274278875,
                              0.010691103432292251,
                              0.005152092120287733, 0.008154024707637389, 0.00750703234766864, 0.01170998288515426,
                              0.006753044359260664,
                              0.010482790225067517, 0.008798076057298295, 0.005388532307637127, 0.006272000048413759,
                              0.008450955040454917,
                              0.010781856319906523, 0.008263574880561878, 0.01018580303103238, 0.005515032368784276,
                              0.007048708633469241,
                              0.0047462660734757375, 0.004997825844321155, 0.00792238558212265, 0.004417622126005647,
                              0.004070862107249665,
                              0.007780570825308247, 0.006868689940640614, 0.0048863447173309515, 0.005131030521956416,
                              0.005133676319820007,
                              0.006997362960493834, 0.0060195208770306196, 0.01025367788301791, 0.00528150655991803,
                              0.00661727754554222,
                              0.009577805416492871, 0.005545692177762029, 0.006767785927764116, 0.008813461588926291,
                              0.009747310289677989,
                              0.007045993877626664, 0.0072478297316020375, 0.009757409746199772, 0.006001374932295513,
                              0.010106846189789735,
                              0.011028998161130451, 0.008444681509354884, 0.007106480020199904, 0.003522165432869641,
                              0.00715167055524964,
                              0.008902730470462616, 0.0047034791486734034, 0.007557583052325103, 0.007583772609179062,
                              0.006638426938325318,
                              0.003585202755380714, 0.012957184957682793, 0.006169309387363514, 0.006998801939184364,
                              0.0032524696608756186]
        plot_experiment_location = "/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Post Filter/Forward_floating_results.png"
        indexes = np.arange(1, 101)
        plt.figure(figsize=(10, 8))
        plt.errorbar(x=indexes, y=cross_validation_scores, yerr=standard_deviation,
                     label='Mean CV scores with their std',
                     marker='_', mfc='red', mec='green', ms=20, mew=4)
        plt.savefig(plot_experiment_location)
    '''
    Fifth Experiment: fix the bugs in the fourth one and plot a good one
    '''
    if experiment_5 not in results_experiments:
        print('Performing experiment 5: vizualisation of features selected via Forward Floating approach')
        cross_validation_scores = [0.2605880881239258, 0.38618817350195656, 0.47516646860152206, 0.54457680544915,
                                   0.584641016256772, 0.6269092700930124,
                                   0.6517334250520718, 0.6664472269497618, 0.6794475642579999, 0.6879273423396581,
                                   0.6954378241031396, 0.703816222588879,
                                   0.7105114477999326, 0.7112334026077232, 0.7143998470553883, 0.7188303797682546,
                                   0.7212846205516111, 0.7222774642450531,
                                   0.722477452761393, 0.7278279957782066, 0.730176811195998, 0.7302927228728077,
                                   0.7324483344754121, 0.735946455043221,
                                   0.7341019770603862, 0.7358582032459523, 0.7372748066111142, 0.7401786248509898,
                                   0.7413357316491861, 0.7411178606160584,
                                   0.7427838809281662, 0.7440129620929467, 0.7417250576627161, 0.7415822997697661,
                                   0.7412468938880519, 0.7444931438878157,
                                   0.7455007341586949, 0.7475769119212243, 0.7493698267927265, 0.7478214963001801,
                                   0.7476535598048549, 0.7452158738358223,
                                   0.7466226136102662, 0.7461510384634461, 0.7455846017966051, 0.7451708765515445,
                                   0.7466004016139246, 0.7431497773967927,
                                   0.7434832483538779, 0.7488982251514892, 0.7473263436827157, 0.7476183626384257,
                                   0.7490976105339975, 0.7495987847726786,
                                   0.7487818594958505, 0.7503178199079236, 0.7487971886454607, 0.7532361462359498,
                                   0.7513196559100241, 0.7504789530805912,
                                   0.748788216790008, 0.7475938368791839, 0.7496201478229659, 0.7505372714458763,
                                   0.7501522401316767, 0.7492028206418585,
                                   0.750663311754589, 0.7495575298906516, 0.7506260878233322, 0.74887544961518,
                                   0.7493869960578481, 0.7500506368803596,
                                   0.7501316105656425, 0.7485643696449982, 0.7492889022618874, 0.7487360046792212,
                                   0.7497624952775667, 0.7483413857814476,
                                   0.7491088327704477, 0.7503986854598108, 0.749561894148742, 0.7500760047720902,
                                   0.7498806780915684, 0.7496657102162843,
                                   0.7478957753962191, 0.74987002433905, 0.7482063751345315, 0.7477604582020358,
                                   0.7491336292512233, 0.7480465021695709,
                                   0.7480766087651078, 0.7468028461411634, 0.7466696203988844, 0.7482709858384079,
                                   0.745707873612819, 0.7464248115916523,
                                   0.7459359521365627, 0.7451616768939654, 0.7447073795702562, 0.7463802158170271]
        standard_deviation = [0.012523124426808833, 0.004040476446754646, 0.013753873080876066, 0.01662823693965504,
                              0.01512294306836487,
                              0.013166405134717956, 0.016962956480547877, 0.01461450161669251, 0.008672379921293868,
                              0.008278583954772493,
                              0.014162860517803879, 0.013178307923928984, 0.012835418495047057, 0.00960589661317563,
                              0.012718471508359232,
                              0.008337097046599554, 0.01188360277439471, 0.009736400706559983, 0.013612493527336303,
                              0.008608799994107446,
                              0.011965093691188568, 0.010221940980042112, 0.012702337469054095, 0.019427836742485948,
                              0.014430414575818673,
                              0.012495416148899685, 0.01601751704874803, 0.010167251599964714, 0.013297227528637292,
                              0.007476990021485804,
                              0.011483572519772386, 0.01391039385200184, 0.0077008356459023425, 0.01103703800708456,
                              0.0070339081107677615,
                              0.011118780143293669, 0.013219453259212558, 0.011598772163175801, 0.014848915176134032,
                              0.008178278598825408,
                              0.008057045749464161, 0.008476916813611927, 0.008380716241103576, 0.008288355274278875,
                              0.010691103432292251,
                              0.005152092120287733, 0.008154024707637389, 0.00750703234766864, 0.01170998288515426,
                              0.006753044359260664,
                              0.010482790225067517, 0.008798076057298295, 0.005388532307637127, 0.006272000048413759,
                              0.008450955040454917,
                              0.010781856319906523, 0.008263574880561878, 0.01018580303103238, 0.005515032368784276,
                              0.007048708633469241,
                              0.0047462660734757375, 0.004997825844321155, 0.00792238558212265, 0.004417622126005647,
                              0.004070862107249665,
                              0.007780570825308247, 0.006868689940640614, 0.0048863447173309515, 0.005131030521956416,
                              0.005133676319820007,
                              0.006997362960493834, 0.0060195208770306196, 0.01025367788301791, 0.00528150655991803,
                              0.00661727754554222,
                              0.009577805416492871, 0.005545692177762029, 0.006767785927764116, 0.008813461588926291,
                              0.009747310289677989,
                              0.007045993877626664, 0.0072478297316020375, 0.009757409746199772, 0.006001374932295513,
                              0.010106846189789735,
                              0.011028998161130451, 0.008444681509354884, 0.007106480020199904, 0.003522165432869641,
                              0.00715167055524964,
                              0.008902730470462616, 0.0047034791486734034, 0.007557583052325103, 0.007583772609179062,
                              0.006638426938325318,
                              0.003585202755380714, 0.012957184957682793, 0.006169309387363514, 0.006998801939184364,
                              0.0032524696608756186]
        plot_experiment_location = "/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Post Filter/Forward_floating_results.png"
        indexes = np.arange(1, 101)
        plt.figure(figsize=(10, 8))
        plt.xlabel('Number of features selected')
        plt.ylabel('Mean Cross Validation Score')
        plt.title('Performances of the Forward Floating feature selection performed on the Filtered Features')
        plt.errorbar(x=indexes, y=cross_validation_scores, yerr=standard_deviation,
                     label='Mean CV scores with their std', mec='green', ms=20, mew=4)
        plt.savefig(plot_experiment_location)
    '''
    Sixth Experiment: Analyzing the features selected by the floating forward approach done on the whole features. 
    '''
    if experiment_6 not in results_experiments:
        print('Experiment 6: Analysis of the features selected by the FF RFE approach on the whole features')
        features_final = pd.read_csv("/home/david/Utils/Experiments/Final/Experiment_number_5/Features.csv")
        features_final.dropna(inplace=True)
        features_final.drop_duplicates(keep='first', inplace=True)
        features_final = features_final.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        features_final = features_final.replace([float('inf'), -float('inf')], np.nan).dropna(axis=0)
        indexes_selected = [2, 15, 17, 26, 29, 31, 45, 46, 51, 76, 92, 122, 124, 145, 170, 180, 205, 231, 232, 240, 252,
                            256, 278, 335, 349, 418, 444, 466, 491, 495, 497, 500, 520, 523, 530, 533, 539, 548, 550,
                            552, 572, 634, 658, 673, 675, 676, 677, 691, 703, 718, 723, 730, 735, 757, 762, 764, 773,
                            795, 801, 844, 854, 892, 897, 900]
        names_selected = features_final.columns[indexes_selected]
        _, repartition_features = np.asarray(repartition(names_selected.values))
        features_experiment_floating = "/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Features.csv"
        results_floating = "/home/david/Utils/Experiments/drift/Experiment 0.5 Hz/Selection/Results_forward_floating_whole_feature.csv"
        data = pd.read_csv(features_experiment_floating)
        data.dropna(inplace=True)
        features_selected = features_final.columns[indexes_selected].values
        features_selected = np.append(features_selected, 'Label')
        print(features_selected)
        print(data.columns)
        with open(results_floating, 'w', newline='') as file_floating:
            writer_floating = csv.writer(file_floating)
            writer_floating.writerow(
                ['Sixth Experiment: Analyzing the features selected by the floating forward approach done'
                 ' on the whole features.'])
            writer_floating.writerow(['Indexes of features selected:', indexes_selected])
            writer_floating.writerow(['Repartition of the features selected:', repartition_features])
            writer_floating.writerow(['Performances of this model with Grid Search CV:'])
        BuildClassifier(data[features_selected], results_floating, search='gs')
    '''
    Seventh Experiment: Analyzing the performances of the new features extracted. 
    '''
    if experiment_7 not in results_experiments:
        print('Experiment 7: Analysis of the new features extracted with the challenge function')
        features_new = pd.read_csv("/home/david/Utils/Experiments/New_Features/Experiment_new_features/Features_challenge.csv")
        features_new.dropna(inplace=True)
        features_new.drop_duplicates(keep='first', inplace=True)
        features_new = features_new.replace([np.inf, -np.inf], 0).dropna(axis=0)
        features_new = features_new.replace([float('inf'), -float('inf')], 0).dropna(axis=0)
        print(len(features_new))
        results_new = '/home/david/Utils/Experiments/New_Features/Experiment_new_features/Results_challenge.csv'
        with open(results_new, 'w', newline='') as file_new:
            writer_new = csv.writer(file_new)
            writer_new.writerow(
                ['Seventh Experiment: Analyzing the performances of the new features extracted, without any FS. '
                 'Later on, we will need to perform Feature Selection with those features'])
            writer_new.writerow(['Performances of this model with Grid Search CV:'])
        BuildClassifier(features_new, results_new, search='gs')
    '''
    Eighth Experiment: Vizualisation of Overfitting for RF model
    '''
    """
    Ninth Experiment: Analyzing the results of the RFE method on the challenge features. The RFE executed was floating forward. 
    The result locaton are: "/home/david/Utils/Experiments/New_Features/Experiment_new_features/Results_fs.csv"
    The indexes selected were: []
    """
    if experiment_9 not in results_experiments:
        features_challenge = pd.read_csv("/home/david/Utils/Experiments/New_Features/Experiment_new_features/Features_challenge.csv")
        indexes_selected_challenge = []
        features_selected = features_challenge.columns[indexes_selected_challenge].index
        features_selected = np.append(features_selected, 'Label')
        features_challenge.dropna(inplace=True)
        features_challenge.drop_duplicates(keep='first', inplace=True)
        features_challenge = features_challenge.replace([np.inf, -np.inf], 0).dropna(axis=0)
        features_challenge = features_challenge.replace([float('inf'), -float('inf')], 0).dropna(axis=0)
        results_new = "/home/david/Utils/Experiments/New_Features/Experiment_new_features/Results_fs.csv"
        with open(results_new, 'a+', newline='') as file_new:
            writer_new = csv.writer(file_new)
            writer_new.writerow(
                ['Ninth Experiment: Analyzing the results of the RFE method on the challenge features. The RFE executed was floating forward.'])
            writer_new.writerow(['Performances of this model with Grid Search CV:'])
        BuildClassifier(features_challenge, results_new, search='gs')





