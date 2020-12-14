import warnings
import numpy as np
import pandas as pd
import os


interesting_keys = ["Pon", "P", "Poff", "QRSon", "Q", "R", "S", "QRSoff", "Ton", "T", "Toff"]


def notching(ecg_signal, on, off):
    """
    Find local extrema in a signal between indexes on and off
    Input: a lead of an ecg, integer indexes of start and end of research
    Output: the indexes of local extrema
    """
    cD = np.diff(ecg_signal)
    notch = []
    for i in range(on + 1, off - 2):
        if ((cD[i] > 0) and (cD[i + 1] < 0)) or ((cD[i] < 0) and (cD[i + 1] > 0)):
            notch.append(i)
    return notch


def find_notching(ecg, fiducial_points, fs):
    """
    This function finds the three consecutive indexes where notching happens, according to the approach of
    Automatic Detection of Strict Left Bundle Branch Block, Radovan Smisek , Pavel Jurak , Ivo Viscor, Josef Halamek ,
    Filip Plesinger , Magdalena Matejkova, Pavel Leinveber , and Jana Kolarova
    """
    indexes_results = []
    QRSon = fiducial_points['QRSon']
    duration = fiducial_points['QRSd']
    start = int(QRSon + 0.04 * int(fs))
    end = int(2 / 3 * duration + QRSon)
    notching_points = notching(ecg, start, end)  # indexes of local extrema in the portion to be studied
    if len(notching_points) > 2:
        ecg_notching = ecg[notching_points] - ecg[QRSon]
        ecg_notchings = [ecg_notching[i:i + 3] for i in range(len(ecg_notching) - 2)]
        trio_selected = [i for i, sublist in enumerate(ecg_notchings) if np.sign(min(sublist)) == np.sign(max(sublist))]
        if len(trio_selected) > 0:
            ecg_notchings = ecg_notchings[trio_selected[0]]
            indexes_results = [i for i, sublist in enumerate(ecg_notchings) if np.argmax(np.abs(sublist)) == 0]
        else:
            return []
        return notching_points[indexes_results[0]: indexes_results[0] + 3]
    else:
        return indexes_results


def compute_features_notching(ecg, fiducial_points, fs):
    """
    The features computed in Automatic Detection of Strict Left Bundle Branch Block
    Radovan Smisek , Pavel Jurak , Ivo Viscor, Josef Halamek , Filip Plesinger , Magdalena Matejkova, Pavel Leinveber ,
    and Jana Kolarova when we spot a notching in the qrs
    """
    indexes_notching = find_notching(np.asarray(ecg), fiducial_points, fs)
    if len(indexes_notching) > 0:
        lengths = np.diff(indexes_notching)
        depths = np.diff(np.asarray(ecg)[indexes_notching])
        begin_area = np.argmin(np.abs(ecg[indexes_notching[0]:indexes_notching[1]] - ecg[indexes_notching[2]])) + \
                     indexes_notching[0]
        area = (indexes_notching[2] - begin_area) * ecg[begin_area] - np.sum(ecg[begin_area:indexes_notching[2]])
        features_notching = pd.DataFrame({'Notch_width_1': [lengths[0]],
                                          'Notch_width_2': [lengths[1]],
                                          'Notch_depth_1': [depths[0]],
                                          'Notch_depth_2': [depths[1]],
                                          'Area_under_the_Notch': [area]
                                          })
    else:
        features_notching = pd.DataFrame({'Notch_width_1': [0],
                                          'Notch_width_2': [0],
                                          'Notch_depth_1': [0],
                                          'Notch_depth_2': [0],
                                          'Area_under_the_Notch': [0]
                                          })
    return features_notching


def windows(signal, length, fs, beginning, end):
    resulting_signal = []
    offset_index = int(length * fs)
    for i in range(beginning, end - offset_index):
        resulting_signal.append(signal[i: i + offset_index + 1])
    return resulting_signal


def compute_features_slurring(ecg, fiducial_points, fs):
    """
    The features computed for slurring in Automatic Detection of Strict Left Bundle Branch Block
    Radovan Smisek , Pavel Jurak , Ivo Viscor, Josef Halamek , Filip Plesinger , Magdalena Matejkova, Pavel Leinveber ,
    and Jana Kolarova when we spot a notching in the qrs
    """
    notching_indexes = find_notching(np.asarray(ecg), fiducial_points, fs)
    if len(notching_indexes) > 0:
        diff = np.diff(ecg)
        portion_1 = diff[notching_indexes[0]: notching_indexes[1]]
        maximum1 = np.max(portion_1) / np.min(portion_1)
        portion_2 = diff[notching_indexes[1]: notching_indexes[2]]
        maximum2 = np.max(portion_2) / np.min(portion_2)
        if notching_indexes[1] - notching_indexes[0] > int(0.012 * 500):
            slices1 = windows(signal=diff, length=0.012, fs=500, beginning=notching_indexes[0],
                              end=notching_indexes[1])
            sum_slices1 = [np.sum(sublist) for sublist in slices1]
            ratio1 = np.max(sum_slices1) / np.min(sum_slices1)
            diff1 = np.max(np.abs(sum_slices1)) - np.min(np.abs(sum_slices1))
        else:
            ratio1 = 0
            diff1 = 0
        if notching_indexes[2] - notching_indexes[1] > int(0.012 * 500):
            slices2 = windows(signal=diff, length=0.012, fs=500, beginning=notching_indexes[1],
                              end=notching_indexes[2])
            sum_slices2 = [np.sum(sublist) for sublist in slices2]
            ratio2 = np.abs(np.max(sum_slices2) / np.min(sum_slices2))
            diff2 = np.max(np.abs(sum_slices2)) - np.min(np.abs(sum_slices2))
        else:
            ratio2 = 0
            diff2 = 0
        if notching_indexes[1] - notching_indexes[0] > int(0.02 * 500):
            slices3 = windows(signal=diff, length=0.02, fs=500, beginning=notching_indexes[0],
                              end=notching_indexes[1])
            sum_slices3 = [np.sum(sublist) for sublist in slices3]
            ratio3 = np.abs(np.max(sum_slices3) / np.min(sum_slices3))
            diff3 = np.max(np.abs(sum_slices3)) - np.min(np.abs(sum_slices3))
        else:
            ratio3 = 0
            diff3 = 0
        if notching_indexes[2] - notching_indexes[1] > int(0.02 * 500):
            slices4 = windows(signal=diff, length=0.02, fs=500, beginning=notching_indexes[1],
                              end=notching_indexes[2])
            sum_slices4 = [np.sum(sublist) for sublist in slices4]
            ratio4 = np.abs(np.max(sum_slices4) / np.min(sum_slices4))
            diff4 = np.max(np.abs(sum_slices4)) - np.min(np.abs(sum_slices4))
        else:
            ratio4 = 0
            diff4 = 0
        features_slurring = pd.DataFrame({'Maximum_1': [maximum1],
                                          'Maximum_2': [maximum2],
                                          'Ratio_1': [ratio1],
                                          'Diff_1': [diff1],
                                          'Ratio_2': [ratio2],
                                          'Diff_2': [diff2],
                                          'Ratio_3': [ratio3],
                                          'Diff_3': [diff3],
                                          'Ratio_4': [ratio4],
                                          'Diff_4': [diff4]
                                          })
    else:
        features_slurring = pd.DataFrame({'Maximum_1': [0],
                                          'Maximum_2': [0],
                                          'Ratio_1': [0],
                                          'Diff_1': [0],
                                          'Ratio_2': [0],
                                          'Diff_2': [0],
                                          'Ratio_3': [0],
                                          'Diff_3': [0],
                                          'Ratio_4': [0],
                                          'Diff_4': [0]
                                          })
    return features_slurring


def extraction_feature_wavedet_BBB(filename, ecg, freq, runtime, list_lead, wavedet_3D_dict, qrs_peak=[]):
    signal_DataFrame = pd.DataFrame()
    list_lead_BBB = [0, 4, 6, 7, 10, 11]
    for i, lead in enumerate(list_lead_BBB):
        feat_lead = pd.DataFrame({'Notch_width_1': [0],
                                  'Notch_width_2': [0],
                                  'Notch_depth_1': [0],
                                  'Notch_depth_2': [0],
                                  'Area_under_the_Notch': [0],
                                  'Maximum_1': [0],
                                  'Maximum_2': [0],
                                  'Ratio_1': [0],
                                  'Diff_1': [0],
                                  'Ratio_2': [0],
                                  'Diff_2': [0],
                                  'Ratio_3': [0],
                                  'Diff_3': [0],
                                  'Ratio_4': [0],
                                  'Diff_4': [0],
                                  'Main_deflection': [0]
                                  })
        wavedet_3D_dict_i = wavedet_3D_dict[i]
        qrs_points = np.asarray(wavedet_3D_dict_i['R'], dtype=np.int32)
        qrs_points = qrs_points[qrs_points > 0]
        if len(qrs_points) > 2:
            try:
                representative_qrs = runtime.FECGSYN_tgen(ecg[lead].tolist(), qrs_points.tolist(), [500])
                representative_qrs = np.squeeze(np.array(representative_qrs['avg']))
                if len(representative_qrs) > 0:
                    representative_qrs = np.squeeze(representative_qrs)
                    template_dict = wavedet_3D_wrapper_template(runtime=runtime, ecg=representative_qrs.tolist() * 3)
                    representative_dict = {key: np.array(value - len(representative_qrs)) for key, value in
                                           template_dict.items() if interesting_keys.__contains__(key)}
                    if representative_dict['QRSon'].size > 0 and representative_dict['QRSoff'].size > 0:
                        representative_dict['QRSon'] = representative_dict['QRSon'][representative_dict['QRSon'] > 0]
                        representative_dict['QRSoff'] = representative_dict['QRSoff'][representative_dict['QRSoff'] > 0]
                        if representative_dict['QRSon'].size > 0 and representative_dict['QRSoff'].size > 0:
                            representative_dict['QRSon'] = representative_dict['QRSon'][0]
                            representative_dict['QRSoff'] = representative_dict['QRSoff'][0]
                            duration = representative_dict['QRSoff'] - representative_dict['QRSon']
                            representative_dict['QRSd'] = np.asarray(duration)
                            feat_lead = extraction_feature_BBB(representative_qrs, lead, freq, representative_dict)

            except:
                print('We handled the exception: nan returned')

        signal_DataFrame = pd.concat([signal_DataFrame, feat_lead], sort=False, axis=1)
    return signal_DataFrame


def main_deflection_qrs(ecg, fiducial_points):
    deflection = 0
    on = fiducial_points['QRSon']
    off = fiducial_points['QRSoff']
    if off > on:
        main_pos = np.argmax(np.abs(ecg[int(on):int(off)]))
        if np.asarray(ecg)[main_pos + on] < 0:
            deflection = 1
    features = pd.DataFrame({'Main_deflection': [deflection]})
    return features


def extraction_feature_BBB(representative_ecg, lead, freq,
                           template_dict, preprocess=False):
    if lead in [0, 4, 6, 7, 10, 11]:
        feat = pd.concat([compute_features_notching(representative_ecg, template_dict, freq),
                          compute_features_slurring(representative_ecg, template_dict, freq)], axis=1)
        feat = pd.concat([feat, main_deflection_qrs(representative_ecg, template_dict)], axis=1)
    else:
        feat = pd.DataFrame()
    return feat


def wavedet_3D_wrapper_template(runtime, ecg):
    ret_val = runtime.python_wrapper_template(ecg)
    for key in (ret_val.keys()):
        if interesting_keys.__contains__(key):
            if ret_val[key] != ret_val[key]:
                ret_val[key] = 0
            ret_val[key] = np.array(ret_val[key], dtype=np.int64)
    return ret_val


def wavedet_3D_wrapper(filename, list_lead, qrs_peak, runtime, ecg):  # engine does not support structure array
    ecg_matlab = []
    for i in range(len(ecg)):
        ecg_matlab.append(ecg[i].tolist())
    ret_val = runtime.python_wrap_wavedet_3D('', ecg_matlab, filename, list_lead, [],
                                         nargout=12)
    ret_val_matlab = [dict(ret_val[i]) for i in range(len(ret_val))]
    for i, lead in enumerate(list_lead):
        for key in (ret_val[i].keys()):
            if interesting_keys.__contains__(key):
                ret_val[i][key] = np.array(ret_val[i][key]._data).reshape(ret_val[i][key].size, order='F')[0]
                ret_val[i][key][np.isnan(ret_val[i][key])] = 0
                ret_val[i][key] = np.asarray(ret_val[i][key][1:], dtype=np.int64)
        R_ref = ret_val[i]['R']
        Poff = ret_val[i]['Poff']
        Ton = ret_val[i]['Ton']
        if interesting_keys.__contains__('Q'):
            for p, ind in enumerate(ret_val[i]['QRSon']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['Q'][p] == 0:
                        if R_ref[p] > ind + 1:
                            candidate = np.argmin(
                                ecg_lead[ind:R_ref[p]]) + ind  # todo: à remplacer par compute_argmin?
                            if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                ref_value = np.max(np.abs(ecg_lead[ind:ind + 5]))

                                if Poff[p] > 0:
                                    indice_minimal = (candidate + Poff[p]) / 2
                                else:
                                    indice_minimal = 0
                                while ind > 0 and np.abs((ecg_lead[
                                                              ind] - ref_value) / ref_value) < 0.2 and ind > indice_minimal:  # todo: divide by zero encountered
                                    ind = ind - 1
                                ret_val[i]['QRSon'][p] = ind
                            ret_val[i]['Q'][p] = candidate  # in order to correct when we do not detect any Q points
                        else:
                            continue
        if interesting_keys.__contains__('S'):
            for p, ind in enumerate(ret_val[i]['QRSoff']):
                ecg_lead = ecg[lead]
                if R_ref[p] > 0:
                    if ind > 0 and ret_val[i]['S'][p] == 0:
                        if ind > R_ref[p] + 1:
                            candidate = np.argmin(ecg_lead[R_ref[p]:ind]) + R_ref[
                                p]  # todo: à remplacer par compute_argmin?
                            if candidate == ind:  # The QRSon point is spotted right at the QRS complex, we will move it while the movement is low
                                ref_value = np.max(np.abs(ecg_lead[ind - 5:ind]))

                                if Ton[p] > 0:
                                    indice_maximal = (candidate + Ton[p]) / 2
                                else:
                                    indice_maximal = len(ecg_lead)
                                while ind < len(ecg_lead) and (
                                        np.abs((ecg_lead[ind] - ref_value) / ref_value)) and ind < indice_maximal:
                                    ind = ind + 1
                                ret_val[i]['QRSoff'][p] = ind
                            ret_val[i]['S'][p] = candidate  # in order to correct when we do not detect any S points
                        else:
                            continue
    return ret_val, ret_val_matlab

