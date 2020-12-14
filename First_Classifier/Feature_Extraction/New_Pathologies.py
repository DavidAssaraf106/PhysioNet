"""
In this file, we are going to conduct experiments in order to see if some of our already implemented features are discriminative between pathologies.
For instance, the RR interval features should help us to discriminate between AF and AFL, which we could not see in classifier performances.
I expect this to be dued to the fact that there are not so many examples of AFL.
"""

import pandas as pd
import matplotlib.pyplot as plt
import csv
import os



if os.name == 'nt':
    input_directory = "C:\\Users\\David\\PhysioNet_Code\\Training_WFDB"

if os.name == 'posix':
    features_location = "/home/david/Utils/Experiments/New_Features/Experiment_new_features/Features_complete_AB.csv"


def boxplot(data, list_features):
    for i in range(len(data.columns)):
        if data.columns[i] in list_features:
            plt.rcParams.update({'font.size': 15})
            data.boxplot(column=data.columns[i], by='Label', grid=False, figsize=(30, 8), showfliers=False)
            plt.xlabel('Class')
            plt.ylabel(str(data.columns[i]) + ' Value')
            plt.savefig('C:\\Users\\David\\PhysioNet\\Documents\\Paper Redaction\\boxplot.png')


if __name__ == '__main__':

    experiment = 'done'
    experiment_J = 'wait'
    experiment_0 = 'done'
    experiment_1 = 'done'
    experiment_2 = 'done'
    experiment_3 = 'done'
    experiment_boxplot = ''
    results_experiments = {'done', 'fail', 'wait'}
    if experiment_J not in results_experiments:
        df = pd.read_csv('/home/david/Ensemble_Learning/Probabilities/Jeremy.csv')
        conversion = {1: 0, 0: 3, 2: 1, 3: 2}
        labels = df.iloc[:, -1]
        labels = [conversion.get(label, label) for i, label in enumerate(labels)]
        df.iloc[:, -1] = labels
        df.to_csv('/home/david/Ensemble_Learning/Probabilities/Jeremy_new.csv')
    if experiment not in results_experiments:
        jeremy_location = '/home/david/Ensemble_Learning/Probabilities/results_082.csv'
        reformatting_jeremy = []
        with open(jeremy_location, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                array = [float(value) for value in row]
                reformatting_jeremy.append(array)
        new_data_jeremy = pd.DataFrame(reformatting_jeremy)
        new_data_jeremy.to_csv('/home/david/Ensemble_Learning/Probabilities/Jeremy.csv')
    if experiment_0 not in results_experiments:
        """
        I wish to see if the features extracted for AF and the RR intervals allow to witness a difference between AF and AFL.
        """
        list_AF = ['cosEn_', 'AFEv_', 'OriginCount_', 'IrrEv_', 'PACEV_', 'AVNN_', 'SDNN_', 'SEM_', 'minRR_', 'medHR_',
                   'PNN20_', 'PNN50_', 'RMSSD_', 'CV_', 'SD1_', 'SD2_',
                   'RR_ratio_max_', 'RR_ratio_median_', 'RR_ratio_mean_', 'RR_ratio_std_', 'IRmax', 'IRmedian',
                   'IRmean', 'IRstd']
        features = pd.read_csv(features_location)
        boxplot(features, list_AF)
    if experiment_1 not in results_experiments:
        """
        Do the same experiment for CRBBB/IRBBB/RBBB
        """
        features = pd.read_csv(features_location)
        list_BBB = ['Notch_width_1',
                    'Notch_width_2',
                    'Notch_depth_1',
                    'Notch_depth_2',
                    'Area_under_the_Notch',
                    'Maximum_1',
                    'Maximum_2',
                    'Ratio_1',
                    'Diff_1',
                    'Ratio_2',
                    'Diff_2',
                    'Ratio_3',
                    'Diff_3',
                    'Ratio_4',
                    'Diff_4',
                    'Main_deflection']
        boxplot(features, list_BBB)
    if experiment_2 not in results_experiments:
        "We are going to try to visualize the differences between AF and AFL in terms of f-wave detection."
        "We know that we are able to distinguish AFL and AF from other pathologies. What remains to be done is "
        "to distinguish AF from AFL."
        features = pd.read_csv(
            "/home/david/Utils/Experiments/New_Features/Experiment_new_features/Features_complete_AB.csv")
        with open('/home/david/Utils/Experiments/New_Features/Experiment_new_features/AFL/experiments_features.csv',
                  'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Those are the experiments we got with ', features['Label'].value_counts()])
        features = features.loc[(features['Label'] == 'AF') | (features['Label'] == 'AFL')]
        boxplot(features, ['max_freq_', 'ratio_'])
    if experiment_3 not in results_experiments:
        features = pd.read_csv('/home/david/Utils/Experiments/New_Features/Experiment_new_features/AFL/frequency_regularity.csv')
        features_AFL = ['Freq_regularity_']
        boxplot(features, features_AFL)
    if experiment_boxplot not in results_experiments:
        data = pd.read_csv('C:\\Users\\David\\PhysioNet\\First_Classifier\\Utils\\Incremental_results\\Features\\features_test_without_SNR.csv')
        boxplot(data, ['RR_ratio_std_7'])
