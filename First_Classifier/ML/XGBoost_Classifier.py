import warnings
import numpy as np
from sklearn.metrics import make_scorer, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import csv
from sklearn.feature_extraction.text import CountVectorizer
import eli5 as eli

pathologies = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
warnings.filterwarnings("ignore")


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


def CrossFoldValidationBoosting(X_train, y_train, search=None):
    scoring_beta = make_scorer(compute_beta_score_CV, greater_is_better=True)
    classifier, random_grid = GB_classifier()
    if search is not None:
        random_model = GridSearchCV(estimator=classifier, param_grid=random_grid, verbose=1, scoring=scoring_beta,
                                    n_jobs=20, return_train_score=True, cv=5, refit=True)
    else:
        random_model = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, verbose=1,
                                          n_jobs=20, scoring=scoring_beta, return_train_score=True, cv=5)
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


def GB_classifier():
    loss = ['deviance', 'exponential']
    learning_rate = np.arange(0, 1, 0.1)
    subsample = np.arange(0, 3, 1)
    criterion = ['friedman_mse', 'mse', 'mae']
    n_estimators = [50, 110, 150, 200]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth_over = [int(x) for x in np.linspace(10, 70, num=5)]
    min_samples_split = [2, 10, 15]
    min_samples_leaf = [1, 4, 5]
    bootstrap = [True, False]
    ccp_alpha = np.arange(0, 0.5, 0.1)
    weighted = ['balanced', 'balanced_subsample']
    random_grid = {'loss': loss,
                   'learning_rate': learning_rate,
                   'subsample': subsample,
                   'criterion': criterion,
                   'ccp_alpha': ccp_alpha,
                   'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth_over,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': weighted
                   }
    gb = GradientBoostingClassifier()
    return gb, random_grid


def explain_classification(model):
    vec = CountVectorizer


def BuildClassifierBoosting(features, result_writing, write=True, search=None):
    X_train, X_test, y_train, y_test = split_data(features)
    X_train, X_test = preprocessing_data(X_train, X_test)
    model_selected = CrossFoldValidationBoosting(X_train, y_train, search)
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
    return model_selected, f_beta


# todo: visualize the predictions of the XGBoost please
if __name__ == '__main__':
    experiment_1 = ''
    results_experiments = {'done', 'fail'}
    '''
    First Experiment: We are going to train a XGBoost model on the Challenge data (the one we extracted with the additional features), 
    and try to visualize the results of this classifier' decisions. 
    We will compare its performances with the ones from the RF on the same dataset
    '''
    if experiment_1 not in results_experiments:
        features_gradient_boosting = pd.read_csv(
            '/home/david/Utils/Experiments/New_Features/Experiment_new_features/Features_challenge.csv')
        results_gradient_boosting = '/home/david/Utils/Experiments/XGBoost/Results_challenge.csv'
        with open(results_gradient_boosting, 'w', newline='') as file_gb:
            writer = csv.writer(file_gb)
            writer.writerow(
                ['First experience: We are going to train a XGBoost model on the Challenge data (the one we extracted '
                 'with the additional features), '
                 'and try to visualize the results of this classifier decisions.'
                 'We will compare its performances with the ones from the RF on the same dataset.'
                 'The scores we had when extracting only challenge features as new features on DataSet A were 0.82 upper side CI and '
                 '0.753 test set'])
            writer.writerow(
                ['We extracted sample from ', len(features_gradient_boosting),
                 'Patients, and Here are the results we got'])
        gradient_boosting = BuildClassifierBoosting(features_gradient_boosting, results_gradient_boosting, search='gs')
