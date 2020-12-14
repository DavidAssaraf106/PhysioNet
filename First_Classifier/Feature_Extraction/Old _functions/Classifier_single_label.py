import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, make_scorer, jaccard_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, FunctionTransformer
import matplotlib.pyplot as plt
from joblib import dump
import mifsmaster.mifs as mifs

import joblib
import os

global_model = True
pathology = 'AF_AVB_PAC_PVC'
pathologies = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']


def load_data():
    if os.name == 'nt':
        if global_model:
            data_file = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Complete_model\\features_" + pathology + '.csv'
        else:
            data_file = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\" + pathology + "\\features_" + pathology + '.csv'
    if os.name == 'posix':
        if global_model:
            data_file = "/home/david/Global_model/features_" + pathology + ".csv"
        else:
            data_file = "/home/david/" + pathology + "/features_" + pathology + ".csv"
    data_new = pd.read_csv(data_file)
    data_new.dropna(inplace=True)
    return data_new


def split_data(features, test_size=0.2):
    label = features['Label']
    features = features.drop(['Label'], axis=1)
    le = LabelEncoder()
    label = le.fit_transform(label)
    return train_test_split(features, label, test_size=0.1, stratify=label)


def preprocessing_data(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    transformer = FunctionTransformer(np.log1p, validate=True)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    print("Size of X_train is:", X_train.shape)
    print("Size of X_test is:", X_test.shape)
    return X_train, X_test


# todo: affiche le validation score, ainsi que sa variance
def CrossFoldValidation(X_train, y_train, model):
    scoring_beta = make_scorer(compute_beta_score_CV, greater_is_better=True)
    if model == "RF":
        print("Using RF classifier for classification of " + pathology)
        classifier, random_grid = RF_classifier()

    # random_model = GridSearchCV(estimator=classifier, param_grid=random_grid, verbose=1, scoring=scoring_beta,
    # n_jobs=-1, return_train_score=True, cv=6, refit=True)

    random_model = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, verbose=1,
                                      n_jobs=-1, scoring=scoring_beta, return_train_score=True, cv=6)

    random_model.fit(X_train, y_train)
    dict_results = random_model.cv_results_

    index = np.argwhere(dict_results['rank_test_score'] == 1)
    index = index[0][0]
    mean_score = dict_results['mean_test_score'][index]
    print(mean_score)
    std_score = dict_results['std_test_score'][index]
    print(std_score)
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

    if plot:
        f_beta_dict = dict(zip(pathologies, fbeta_l))
        g_beta_dict = dict(zip(pathologies, gbeta_l))
        print("f_beta of classes", f_beta_dict)
        print("g_beta of classes", g_beta_dict)
    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return accuracy, f_measure, f_beta, g_beta


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
        _, _, f_beta, g_beta = (compute_beta_score(y_val_beta, y_pred_beta, 2, num_classes=9))
        score = np.sqrt(f_beta * g_beta)
        return score
    else:
        return 1


# todo: change pour afficher bien le test set
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

    accuracy, _, f_beta, g_beta = (
        compute_beta_score(y_test_beta, y_pred_total_beta, 2, num_classes=9,
                           plot=True))  # accuracy, f_measure, f_beta, g_beta
    print("F_beta_score", f_beta)
    print("G_beta_score", g_beta)
    geometric_mean = np.sqrt(f_beta * g_beta)
    print("Geometric mean:", geometric_mean)
    return geometric_mean


def RF_classifier():
    # Number of trees in random forest
    n_estimators = [50, 90, 110, 130, 150, 200, 220, 250, 300]
    n_estimators_over = [50, 90, 110, 130, 150, 200, 220]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=7)]
    max_depth_over = [int(x) for x in np.linspace(10, 70, num=7)]
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


def importance_features(model, data):
    list = data.columns[:-1]
    feature_imp = pd.Series(model.best_estimator_.feature_importances_, index=list).sort_values(ascending=False)
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = 18, 8
    feature_imp.plot(kind='barh')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    if os.name == 'nt':
        if global_model:
            localisations = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Complete_model\\importance_features_" + pathology + '.png'
        else:
            localisations = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\" + pathology + "\\importance_features_" + pathology + '.png'
    if os.name == 'posix':
        if global_model:
            localisations = "/home/david/Global_model/importance_features_" + pathology + ".png"
        else:
            localisations = "/home/david/" + pathology + "/importance_features_" + pathology + ".png"
    plt.savefig(localisations)
    plt.cla()
    plt.clf()
    plt.close()


# todo: print a good confusion matrix, and delete the old one, we have seaborn now
def BuildClassifier(features, model, confusion_matrix=False, importance=False, selection=True):
    X_train, X_test, y_train, y_test = split_data(features)
    X_train, X_test = preprocessing_data(X_train, X_test)
    model_selected = CrossFoldValidation(X_train, y_train, model)
    score = TestSet(X_test, y_test, model_selected)
    if confusion_matrix:
        plot_confusion_matrix(model_selected.best_estimator_, X_test, y_test)
        if os.name == 'nt':
            if global_model:
                data_save = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Complete_model\\confusion_matrix" + pathology + '.png'
            else:
                data_save = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\" + pathology + "\\confusion_matrix" + pathology + '.png'
        if os.name == 'posix':
            if global_model:
                data_save = "/home/david/Global_model/confusion_matrix_" + pathology + ".png"
            else:
                data_save = "/home/david/" + pathology + "/confusion_matrix_" + pathology + ".png"
        plt.savefig(data_save)
        plt.cla()
        plt.clf()
        plt.close()
    if importance:
        importance_features(model_selected, features)
    if selection:
        method = 'MRMR'
        feature_selection_MRMR(features, method)
    return model_selected, score


def feature_selection_MRMR(data, string):
    X = data[data.columns[:-1]]
    y = data['Label']
    le = LabelEncoder()
    label = le.fit_transform(y)
    feat_selector = mifs.MutualInformationFeatureSelector(method=string, verbose=1)
    feat_selector.fit(X, label)
    X_filtered = feat_selector.transform(X)
    print(X.columns[feat_selector._get_support_mask()].values())


def boxplot():  # todo: change names of the features
    if os.name == 'nt':
        if global_model:
            data_file = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Complete_model\\"
        else:
            data_file = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\" + pathology + "\\"
    if os.name == 'posix':
        if global_model:
            data_file = "/home/david/Global_model/"
        else:
            data_file = "/home/david/" + pathology + "/"
    data = pd.read_csv(data_file)
    for i in range(len(data.columns) - 3):
        data.boxplot(column=data.columns[i], by='Label', grid=False)
        plt.savefig(data_file + 'boxplot_' + data.columns[i] + '.png')
        plt.cla()
        plt.clf()
        plt.close()


# BuildClassifier(data, 'RF', confusion_matrix, importance)
if __name__ == '__main__':
    data_new = load_data()
    model_selected, score = BuildClassifier(data_new, "RF", confusion_matrix=True, importance=True, selection=True)
    # boxplot()
