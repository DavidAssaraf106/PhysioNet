import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import wfdb.processing
from scipy.signal import butter, lfilter
import os
import math

# folder where we will be saving the ecg where we do not detect any peaks, or less than 5
path_to_problematic_leads = "C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\Problematic_ECG"

#todo: supprime toutes les fonctions qu'on n'utilise plus
def compute_mean(array):
    if len(array) > 0:
        return np.mean(array)
    else:
        return 0


def compute_median(array):
    if len(array) > 0:
        return np.median(array)
    else:
        return 0


def compute_std(array, ddof=1):
    if len(array) > 1:
        return np.std(array, ddof=ddof)
    else:
        return 0


def minimum(array):  # handles when the array is empty
    if len(array) > 0:
        return np.min(array)
    else:
        return 0


def maximum(array, axis=0):
    if len(array) > 0:
        return np.max(array, axis)
    else:
        return 0


# The functions we need in order to get the gain and the frequence for the filters and the denoising
def get_gain_lead(header_data):
    tmp_hea = header_data[0].split(' ')
    num_leads = int(tmp_hea[1])
    gain_lead = np.zeros(num_leads)
    for ii in range(num_leads):
        tmp_hea = header_data[ii + 1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])
    return gain_lead


def get_freq(header_data):
    for lines in header_data:
        tmp = lines.split(' ')
        return tmp[2]


def integrate(ecg, ws):
    lgth = ecg.shape[0]
    integrate_ecg = np.zeros(lgth)
    ecg = np.pad(ecg, math.ceil(ws / 2), mode='symmetric')
    for i in range(lgth):
        integrate_ecg[i] = np.sum(ecg[i:i + ws]) / ws
    return integrate_ecg


# Three following functions: filter to remove low frequency signals (= noise), find_peaks and the global function to detect R_peaks; already implemented in the baseline model
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def notching(ecg_signal, on, off):
    cD = np.diff(ecg_signal)
    notch = []
    for i in range(on + 1, off - 2):
        if (((cD[i] > 0) and (cD[i + 1] < 0)) or ((cD[i] < 0) and (cD[i + 1] > 0))):
            notch.append(i)
    return notch

# We realized that the function was sometimes spotting wrong the R-peaks, so we needed to shift a bit the locations
def adjusting_R_windows(ecg_measurement_signal, idx):
    refractory_period = 150  # there is at least 0.6s btw 2 R-peaks = 300 indexes
    window = 70  # the window in which we are looking for the "true" R-peaks around the one detected
    res_idx = np.zeros(idx.shape, dtype=int)
    for i in range(len(idx)):
        index = idx[i]
        ind_max = np.argmax(
            ecg_measurement_signal[max(0, index - window):min(index + window, len(ecg_measurement_signal) - 1)])
        if index > window:
            res_idx[i] = ind_max + index - window
        else:
            res_idx[i] = ind_max
    # now, we need to remove peaks that are too close from one another
    res_idx_true = np.zeros(len(res_idx), dtype='bool')
    res_idx_true[:] = True
    for i in range(len(res_idx) - 1):
        if res_idx_true[i] == False:
            continue
        j = i + 1
        ind_ref = res_idx[i]
        while j < len(res_idx) and np.abs(res_idx[j] - ind_ref) < refractory_period:
            res_idx_true[j] = False
            j = j + 1
    return res_idx[res_idx_true]


def find_Q_point(ecg, R_peaks):
    Q_point = []
    for index in range(len(R_peaks)):
        cnt = R_peaks[index]
        if cnt - 1 < 0:
            Q_point.append(0)
            continue
        while ecg[cnt] >= ecg[cnt - 1]:
            cnt -= 1
            if cnt < 0:
                cnt = 0
                break
        Q_point.append(cnt)
    return np.asarray(Q_point)


def adjusting_Q_windows(ecg_measurement_signal, Q_idx):
    window = 40
    idx_Q_bis = np.zeros(Q_idx.shape, dtype=int)
    for i in range(len(Q_idx)):
        index = Q_idx[i]
        if index == 0:
            idx_Q_bis[i] = 0
        else:
            ind_max = np.argmin(ecg_measurement_signal[max(0, index - window):index])
            idx_Q_bis[i] = ind_max + index - window
    return idx_Q_bis


def find_S_point(ecg, R_peaks):
    S_point = []
    for index in range(len(R_peaks)):
        cnt = R_peaks[index]
        if cnt + 1 >= ecg.shape[0]:
            S_point.append(ecg.shape[0] - 1)
            continue
        while ecg[cnt] >= ecg[cnt + 1]:
            cnt += 1
            if cnt + 1 >= ecg.shape[0]:
                break
        S_point.append(cnt)
    return np.asarray(S_point)


# Same remark as we did for the R-peaks: look for the True S_point
def adjusting_S_windows(ecg_measurement_signal, S_idx):
    window = 40
    idx_S_bis = np.zeros(S_idx.shape, dtype=int)
    for i in range(len(S_idx)):
        index = S_idx[i]
        if index >= len(ecg_measurement_signal) - 1:
            idx_S_bis[i] = len(ecg_measurement_signal) - 1
        elif index < 0:
            idx_S_bis[i] = 0
        else:
            ind_max = np.argmin(ecg_measurement_signal[index:min(index + window, len(ecg_measurement_signal) - 1)])
            idx_S_bis[i] = ind_max + index
    return idx_S_bis


def plot_QRS(integrated_ecg, R_points, num_lead, filename):
    directory_file = path_to_problematic_leads + '\\' + filename
    ecg_file = directory_file + '\\lead_' + str(num_lead) + '.png'
    if os.path.exists(ecg_file):
        return
    else:
        fig = plt.figure(figsize=(25, 10))
        plt.ylim(-2000, 2000)
        plt.plot(integrated_ecg)
        plt.plot(np.asarray(R_points), integrated_ecg[R_points], marker="o", ls="", ms=8, color="r", label="R")
        plt.legend()
        plt.title('ECG ' + str(num_lead))
        if os.path.exists(directory_file):
            fig.savefig(ecg_file)
        else:
            os.mkdir(directory_file)
            fig.savefig(ecg_file)


# For these features, we have based our approach on the following papers: (1): Cardiac Arrhythmia Detection By ECG
# Feature Extraction, Rameshwari S Mane, A N Cheeran, Vaibhav D Awandekar, Priya Rani (2): Automatic detection of
# premature atrial contractions in the electrocardiogram Vessela T. Krasteva, Irena I. Jekova, Ivaylo I. Christov (
# 3): Premature Ventricular Contraction (PVC) Detection Using R Signals Brahmantya Aji Pramudita, Anggit Ferdita
# Nugraha, Hanung Adi Nugroho and Noor Akhmad Setiawan (4): Bayesian Classification Models for Premature Ventricular
# Contraction Detection on ECG Traces Félix F. González-Navarro, Jose Antonio Cardenas-Haro


def calculate_S_duration(ecg, R_points, S_points, QRSOn):
    S_duration = np.zeros(len(R_points))
    for i in range(len(R_points)):
        if S_points[i] > R_points[i] + 1:
            isoelectric_line = ecg[QRSOn[i]]
            ind_begin_S_wave = np.argmin(
                np.abs(ecg[R_points[i]: S_points[i]] - isoelectric_line))  # handle the bugs in this function
            S_duration[i] = S_points[i] - ind_begin_S_wave
    S_duration = np.asarray(S_duration)
    return S_duration[S_duration > 0]


def calculate_R_duration(ecg, Q_points, R_points, QRSOn):  # @David, LBBB detection
    R_duration = np.zeros(len(R_points))  # we never select the first one since we can have some negative indexes
    for i in range(1, len(R_points)):
        if R_points[i] > Q_points[i]:
            isoelectric_line = ecg[QRSOn[i]]
            ind_end_R_wave = np.argmin(
                np.abs(ecg[Q_points[i]: R_points[i]] - isoelectric_line))
            R_duration[i] = ind_end_R_wave - Q_points[i]
    R_duration = np.asarray(R_duration)
    return R_duration[R_duration > 0]


def calculate_interbeat(R_points):  # @David, for PAC detection: ok, todo: use for the indexes where we can use RR intervals
    RR = comp_diff(R_points)  # distance between R peaks
    RR_diff = np.asarray(np.asarray(RR[1:]) - np.asarray(RR[:-1]))
    Normalization_tab = (np.asarray(RR[:-6]) + np.asarray(RR[1:-5]) + np.asarray(RR[2:-4]) + np.asarray(
        RR[3:-3]) + np.asarray(RR[4:-2])) / 5
    indexes = Normalization_tab > 0
    RR_diff = RR_diff[5:]
    RR_diff, Normalization_tab = RR_diff[indexes], Normalization_tab[indexes]
    return (RR_diff / np.asarray(Normalization_tab)) * 100


def calculate_QRS_Width(QRSOn, QRSOff):  # @David, for PAC detection, todo: use for indexes effective of QRSon and QRSoff: ok
    QRS_Width = np.asarray(np.asarray(QRSOff) - np.asarray(QRSOn))
    Median_5_Width = []
    for i in range(5, len(QRSOn)):
        Median_5_Width.append(compute_median(QRS_Width[i - 5:i]))
    Median_5_Width = np.asarray(Median_5_Width)
    if len(QRS_Width) < 6:
        return [0]
    return (np.abs(QRS_Width[5:] - Median_5_Width) / Median_5_Width) * 100


def calculate_QRS_Area(ecg, QRSOn, QRSOff):  # @David, for PAC detection
    QRS_Area = np.zeros(len(QRSOn))
    for i in range(len(QRSOn)):
        QRS_Area[i] = np.sum(np.abs(ecg[QRSOn[i]: QRSOff[i]]))
    Median_5_Area = []
    for i in range(5, len(QRSOn)):
        Median_5_Area.append(compute_median(QRS_Area[i - 5:i]))
    Median_5_Area = np.asarray(Median_5_Area)
    if len(QRS_Area) < 6:
        return [0]
    return (np.abs(QRS_Area[5:] - Median_5_Area) / Median_5_Area) * 100


def comp_diff(R_points):  # @Jeremy, for PVC detection
    R_points = np.asarray(R_points)
    cnt_diff_ecg = []
    for idx_q in range(1, len(R_points)):
        cnt_diff = R_points[idx_q] - R_points[idx_q - 1]
        cnt_diff_ecg.append(cnt_diff)
    return cnt_diff_ecg


def energy_peak(ecg, Q_points, S_points):  # @Jeremy, for PVC detection
    cnt_energy_ecg = []
    for idx_q in range(len(Q_points)):
        cnt_energy = 0
        idx_ecg = Q_points[idx_q]
        while idx_ecg < S_points[idx_q]:
            cnt_energy += ecg[idx_ecg] * ecg[idx_ecg]
            idx_ecg += 1
        cnt_energy_ecg.append(cnt_energy)
    return cnt_energy_ecg


def calculate_energy(ecg):  # @Jeremy, for PVC detection
    energy = 0
    for sample in ecg:
        energy += sample * sample
    return energy


def zero_crossing(ecg, Q_points, S_points):  # @Jeremy, for PVC detection
    cnt_crossing_ecg = []
    for idx_q in range(len(Q_points)):
        cnt_crossing = 0
        idx_ecg = Q_points[idx_q]
        while idx_ecg < S_points[idx_q]:
            if (ecg[idx_ecg] > 0) & (ecg[idx_ecg - 2] < 0):
                cnt_crossing += 1
            if (ecg[idx_ecg] < 0) & (ecg[idx_ecg - 2] > 0):
                cnt_crossing += 1
            idx_ecg += 1
        cnt_crossing_ecg.append(cnt_crossing)
    return cnt_crossing_ecg


def calculate_coef_Rwave(ecg, Q_points, S_points):  # @David, for LBBB detection
    indexes = np.zeros(len(Q_points))
    for i in range(len(Q_points)):
        ind_debut = Q_points[i]
        ind_fin = S_points[i]
        ref = ecg[ind_fin]
        indexes[i] = compute_mean(ecg[ind_debut:ind_fin] > ref)
    return compute_mean(indexes)


def Baseline(ecg, array_index_1, array_index_2):
    array_result = []
    for i in range(min(len(array_index_1), len(array_index_2))):
        array_result.append(compute_mean(np.asarray(ecg[array_index_1[i]:array_index_2[i]])))
    return np.asarray(array_result)


def compute_setoff(ecg, array_index, freq, time_setoff):
    array_result = []
    for i in range(len(array_index)):
        array_result.append(ecg[min(array_index[i] + int(time_setoff * int(freq)), len(ecg) - 1)])
    if array_result[-1] == ecg[-1]:
        del array_result[-1]
    return maximum(array_result), minimum(array_result), compute_mean(array_result), compute_median(
        array_result), compute_std(
        array_result)


#  The remaining of the functions are the one used by Armand for AF detection. I have entrusted his functions and the utility of the features he selected.
#  You will find every paper he refers to in every function.
#  In order to run it properly, you need to download edges_hist file and insert the file path in the function metrics.

from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression


def sc_median(data, medfilt_lg=9):
    """ This function implements a median filter used to smooth the spo2 time series and avoid sporadic
        increase/decrease of SpO2 which could affect the detection of the desaturations.
        :arg data: input spo2 time series (!!assumed to be sampled at 1Hz).
        :arg medfilt_lg (optional): median filter length. Default value: 9
        :returns data_med: the filtered data."""

    data_med = signal.medfilt(np.round(data), medfilt_lg)

    return data_med


def sc_resamp(data, fs):
    """ This function is used to re-sample the data at 1Hz. It takes the median SpO2 value
        over each window of length fs so that the resulting output signal is sampled at 1Hz.
        Wrapper of the scipy.signal.resample function
        :arg data: Input SpO2 time series.
        :arg fs: Sampling frequency of the original time series (Hz).
        :returns data_out: The re-sampled SpO2 time series at 1 [Hz].
    """

    data_out = signal.resample(data, int(len(data) / fs))
    return data_out


def sc_desaturations(data, thres=3):
    """
    This function implements the algorithm of:
      Hwang, Su Hwan, et al. "Real-time automatic apneic event detection using nocturnal pulse oximetry."
      IEEE Transactions on Biomedical Engineering 65.3 (2018): 706-712.
    NOTE: The original function search desaturations that are minimum 10 seconds long and maximum 90 seconds long.
    In addition the original algorithm actually looked to me more like an estimate of the ODI4 than ODI3.
    This implementation is updated to allow the estimation of ODI3 and allows desaturations that are up to 120 seconds
    based on some of our observations. In addition, some conditions were added to avoid becoming blocked in infinite while
    loops.
    Important: The algorithm assumes a sampling rate of 1Hz and a quantization of 1% to the input data.
    :param data: SpO2 time series sampled at 1Hz and with a quantization of 1%.
    :param thres: Desaturation threshold below 'a' point (default 2%). IMPORTANT NOTE: 2% below 'a' corresponds to a 3% desaturation.
    :return table_desat_aa:  Location of the aa feature points (beginning of the desaturations).
    :return table_desat_bb:  Location of the aa feature points (lowest point of the desaturations).
    :return table_desat_cc:  Location of the aa feature points (end of the desaturations).
    """
    aa = 1
    bb = 0
    cc = 0
    out_b = 0
    out_c = 0
    desat = 0
    max_desat_lg = 120  # was 90 sec in the original paper. Changed to 120 because I have seen longer desaturations.
    lg_dat = len(data)
    table_desat_aa = []
    table_desat_bb = []
    table_desat_cc = []

    while aa < lg_dat:
        if aa + 10 > lg_dat:  # added condition to test that between aa and the end of the recording there is at least 10 seconds
            return desat, table_desat_aa, table_desat_bb, table_desat_cc

        if data[aa] > 25 and (data[aa] - data[aa - 1]) <= -1 and -thres <= (data[aa] - data[aa - 1]):
            bb = aa + 1
            out_b = 0

            while bb < lg_dat and out_b == 0:
                if bb == lg_dat - 1:  # added this condition in case cc is never reached at the end of the recording
                    return desat, table_desat_aa, table_desat_bb, table_desat_cc

                if data[bb] <= data[bb - 1]:
                    if data[aa] - data[bb] >= thres:
                        cc = bb + 1

                        if cc >= lg_dat:
                            # this is added to stop the loop when c has reached the end of the record
                            return desat, table_desat_aa, table_desat_bb, table_desat_cc
                        else:
                            out_c = 0

                        while cc < lg_dat and out_c == 0:
                            if ((data[aa] - data[cc]) <= 1 or (data[cc] - data[bb]) >= thres) and cc - aa >= 10:
                                if cc - aa <= max_desat_lg:
                                    desat = desat + 1
                                    table_desat_aa = np.append(table_desat_aa, [aa])
                                    table_desat_bb = np.append(table_desat_bb, [bb])
                                    table_desat_cc = np.append(table_desat_cc, [cc])
                                    aa = cc + 1
                                    out_b = 1
                                    out_c = 1
                                else:
                                    aa = cc + 1
                                    out_b = 1
                                    out_c = 1
                            else:
                                cc = cc + 1
                                if cc > lg_dat - 1:
                                    return desat, table_desat_aa, table_desat_bb, table_desat_cc

                                if data[bb] >= data[cc - 1]:
                                    bb = cc - 1
                                    out_c = 0
                                else:
                                    out_c = 0
                    else:
                        bb = bb + 1

                else:
                    aa = aa + 1
                    out_b = 1
        else:
            aa = aa + 1

    return desat, table_desat_aa, table_desat_bb, table_desat_cc


def bsqi(refqrs, testqrs, agw=0.05, fs=200):
    """
    This function is based on the following paper:
        Li, Qiao, Roger G. Mark, and Gari D. Clifford.
        "Robust heart rate estimation from multiple asynchronous noisy sources
        using signal quality indices and a Kalman filter."
        Physiological measurement 29.1 (2007): 15.
    The implementation itself is based on:
        Behar, J., Oster, J., Li, Q., & Clifford, G. D. (2013).
        ECG signal quality during arrhythmia and its application to false alarm reduction.
        IEEE transactions on biomedical engineering, 60(6), 1660-1666.
    :param refqrs:  Annotation of the reference peak detector (Indices of the peaks).
    :param testqrs: Annotation of the test peak detector (Indices of the peaks).
    :param agw:     Agreement window size (in seconds)
    :param fs:      Sampling frquency [Hz]
    :returns F1:    The 'bsqi' score, between 0 and 1.
    """

    agw *= fs
    if len(refqrs) > 0 and len(testqrs) > 0:
        NB_REF = len(refqrs)
        NB_TEST = len(testqrs)

        tree = cKDTree(refqrs.reshape(-1, 1))
        Dist, IndMatch = tree.query(testqrs.reshape(-1, 1))
        IndMatchInWindow = IndMatch[Dist < agw]
        NB_MATCH_UNIQUE = len(np.unique(IndMatchInWindow))
        TP = NB_MATCH_UNIQUE
        FN = NB_REF - TP
        FP = NB_TEST - TP
        Se = TP / (TP + FN)
        PPV = TP / (FP + TP)
        if (Se + PPV) > 0:
            F1 = 2 * Se * PPV / (Se + PPV)
            _, ind_plop = np.unique(IndMatchInWindow, return_index=True)
            Dist_thres = np.where(Dist < agw)[0]
            meanDist = compute_mean(Dist[Dist_thres[ind_plop]]) / fs
        else:
            return 0

    else:
        F1 = 0
        IndMatch = []
        meanDist = fs
    return F1


def comp_dRR(data):
    """
    This function computes the differences of successive RR intervals.
    :param data:    The RR interval input window.
    :returns dRR_s: The RR differences time series.
    """
    # RR interval must be received in seconds
    RR_s = np.vstack((data[1:], data[:-1])).transpose().astype(float)
    dRR_s = np.zeros(RR_s.shape[0])

    # Normalization factors (normalize according to the heart rate)
    k1 = 2
    k2 = 0.5
    mask_low = np.sum(RR_s < 0.5, axis=1) >= 1
    mask_high = np.sum(RR_s > 1, axis=1) >= 1
    mask_other = np.logical_not(np.logical_or(mask_low, mask_high))
    dRR_s[mask_other] = (RR_s[mask_other, 0] - RR_s[mask_other, 1])
    dRR_s[mask_high] = k2 * (RR_s[mask_high, 0] - RR_s[mask_high, 1])
    dRR_s[mask_low] = k1 * (RR_s[mask_low, 0] - RR_s[mask_low, 1])
    return dRR_s


def BPcount(sZ):
    """ Helper function for the computation of the AFEv feature.
        Computes the center bin counts of a partial 15x15 window belogning to the AFEv histogram.
        Cleans out the center bin counts.
    :param sZ:      The input 15x15 matrix.
    :returns BC:    The number of non-zero bins in the histogram.
    :returns PC:    The number of points present in the non-zero bins in the histogram.
    :returns sZ:    The input matrix while the main diagonal and the 4 main side diagonals are cancelled out.
    """
    BC = 0
    PC = 0

    for i in range(-2, 3):
        bdc = np.sum(np.diag(sZ, i) != 0)
        pdc = np.sum(np.diag(sZ, i))
        BC = BC + bdc
        PC = PC + pdc
        sZ = sZ - np.diag(np.diag(sZ, i), i)

    return BC, PC, sZ


def metrics(dRR):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param dRR:     The successive RR differences.
    :returns OriginCount:   The number of points in the center bin (Indicator of Normal Sinus Rhythm).
    :returns IrrEv:         The IrrEv metric as described in the paper (Indicator of Heart Rate Irregularities).
    :returns PACEv:         The PACEv metric as described in the paper (Indicator of Ectopic Beats).
    """

    dRR = np.vstack((dRR[1:], dRR[:-1])).transpose().astype(float)
    # COMPUTE OriginCount
    OCmask = 0.02
    ol = np.sum(np.abs(dRR) <= OCmask, axis=1)
    OriginCount = np.sum(ol == 2)

    # DELETE OUTLIERS | dRR | >= 1.5
    OLmask = 1.5
    dRRnew = dRR[np.sum(np.abs(dRR) >= OLmask, axis=1) == 0, :]

    if dRRnew.size == 0:
        dRRnew = np.array([0, 0]).reshape((1, 2))

    # BUILD HISTOGRAM
    # Specify bin centers of the histogram
    # insert your path
    if os.name == 'nt':
        bin_c = sio.loadmat(str("C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\edges_hist.mat"))['edges'][0][0][
            0]  # Used since there were precision differences between matlab and python.
    if os.name == 'posix':
        bin_c = \
        sio.loadmat(str("/home/david/Utils/edges_hist.mat"))['edges'][0][0][
            0]  # Used since there were precision differences between matlab and python.
    bin_c[0] = -np.inf
    bin_c[-1] = np.inf

    # Three dimensional histogram of bivariate data - 30x30 matrix
    Z, _, _ = np.histogram2d(dRRnew[:, 0], dRRnew[:, 1], bins=(bin_c, bin_c))

    # Clear SegmentZero
    Z[13, 14:16] = 0
    Z[14:16, 13:17] = 0
    Z[16, 14:16] = 0

    # COMPUTE BinCount12
    # COMPUTE PointCount12

    # Z2 contains all the bins belonging to the II quadrant of Z
    Z2 = Z[15:, 15:]
    BC12, PC12, sZ2 = BPcount(Z2)
    Z[15:, 15:] = sZ2

    # COMPUTE BinCount11
    # COMPUTE PointCount11

    # Z3 contains points belonging to the III quadrant of Z
    Z3 = Z[15:, :15]
    Z3 = np.fliplr(Z3)
    BC11, PC11, sZ3 = BPcount(Z3)
    Z[15:, :15] = np.fliplr(sZ3)

    # COMPUTE BinCount10
    # COMPUTE PointCount10

    # Z4 contains points belonging to the IV quadrant of Z
    Z4 = Z[:15, :15]
    BC10, PC10, sZ4 = BPcount(Z4)
    Z[:15, :15] = sZ4

    # COMPUTE BinCount9
    # COMPUTE PointCount9

    # Z1 cointains points belonging to the I quadrant of Z
    Z1 = Z[:15, 15:]
    Z1 = np.fliplr(Z1)
    BC9, PC9, sZ1 = BPcount(Z1)
    Z[:15, 15:] = np.fliplr(sZ1)

    # COMPUTE BinCount5
    BC5 = np.sum(Z[:15, 13:17] != 0)
    # COMPUTE PointCount5
    PC5 = np.sum(Z[:15, 13:17])
    # COMPUTE BinCount7
    BC7 = np.sum(Z[15:, 13:17] != 0)
    # COMPUTE PointCount7
    PC7 = np.sum(Z[15:, 13:17])

    # COMPUTE BinCount6
    BC6 = np.sum(Z[13:17, :15] != 0)
    # Compute PointCount6
    PC6 = np.sum(Z[13:17, :15])

    # COMPUTE BinCount8
    BC8 = np.sum(Z[13:17, 15:] != 0)
    # COMPUTE PointCount8
    PC8 = np.sum(Z[13:17, 15:])

    # CLEAR SEGMENTS 5, 6, 7, 8

    # Clear segments 6 and 8
    Z[13:17, :] = 0
    # Clear segments 5 and 7
    Z[:, 13:17] = 0

    # COMPUTE BinCount2
    BC2 = np.sum(Z[:13, :13] != 0)
    # COMPUTE PointCount2
    PC2 = np.sum(Z[:13, :13])

    # COMPUTE BinCount1
    BC1 = np.sum(Z[:13, 17:] != 0)
    # COMPUTE PointCount1
    PC1 = np.sum(Z[:13, 17:])

    # COMPUTE BinCount3
    BC3 = np.sum(Z[17:, :13] != 0)
    # COMPUTE PointCount3
    PC3 = np.sum(Z[17:, :13])

    # COMPUTE BinCount4
    BC4 = np.sum(Z[17:, 17:] != 0)
    # COMPUTE PointCount4
    PC4 = np.sum(Z[17:, 17:])

    # COMPUTE IrregularityEvidence
    IrrEv = BC1 + BC2 + BC3 + BC4 + BC5 + BC6 + BC7 + BC8 + BC9 + BC10 + BC11 + BC12

    # COMPUTE PACEvidence
    PACEv = (PC1 - BC1) + (PC2 - BC2) + (PC3 - BC3) + (PC4 - BC4) + (PC5 - BC5) + (PC6 - BC6) + (PC10 - BC10) - (
            PC7 - BC7) - (PC8 - BC8) - (PC12 - BC12)

    return OriginCount, IrrEv, PACEv


def comp_sampEn(y, M, r):
    """
    This function implements the algorithm of:
        Richman, Joshua S., and J. Randall Moorman.
        "Physiological time-series analysis using approximate entropy and sample entropy."
        American Journal of Physiology-Heart and Circulatory Physiology 278.6 (2000): H2039-H2049.
    Sample Entropy is an indicator of irregularity in the input signal and hence a good indicator for AF.
    :param y: The input data (RR interval time series)
    :param M: The maximal size of the sub-segments for which the matching is checked.
    :param r: Confidence interval to define matching between two sub-segments.
    :returns e: The sample entropy coefficients for m = 1, ..., M
    :returns A: Number of matching segments of size m
    :returns B: Number of matching segments of size m - 1.
    """
    n = len(y)
    A = np.zeros((M, 1))
    B = np.zeros((M, 1))
    p = np.zeros((M, 1))
    e = np.zeros((M, 1))

    X_A = [np.vstack(tuple(y[i:(len(y) - m + 1 + i)] for i in range(m))).transpose().astype(float) for m in
           range(1, M + 1)]
    len_X_A = np.array([len(x) for x in X_A])
    X_B = [x[:-1, :] for x in X_A]
    len_X_B = len_X_A - 1
    repeated_X_A = [np.repeat(X_A[i], len_X_A[i], axis=0) for i in range(M)]
    tiled_X_A = [np.tile(X_A[i], (len_X_A[i], 1)) for i in range(M)]
    repeated_X_B = [np.repeat(X_B[i], len_X_B[i], axis=0) for i in range(M)]
    tiled_X_B = [np.tile(X_B[i], (len_X_B[i], 1)) for i in range(M)]
    A = np.array([(np.sum(maximum(np.abs(repeated_X_A[i] - tiled_X_A[i]), axis=1) < r) - len_X_A[i]) / 2 for i in
                  range(M)]).reshape(M, 1)
    B = np.array([(np.sum(maximum(np.abs(repeated_X_B[i] - tiled_X_B[i]), axis=1) < r) - len_X_B[i]) / 2 for i in
                  range(M)]).reshape(M, 1)
    N = n * (n - 1) / 2
    if N == 0:
        p[0] = 1
    else:
        p[0] = (A[0] + 1) / N
    e[0] = -np.log(p[0])
    for m in range(1, M):
        p[m] = (A[m] + 1) / (B[m - 1] + 1)  # modified by David in order to prevent 0
        e[m] = -np.log(p[m])
    return e, A, B  # interesting quantity: log(A[m]/B[m-1])


def comp_cosEn(segment):
    """
    This function implements the algorithm of:
        Lake, Douglas E., and J. Randall Moorman.
        "Accurate estimation of entropy in very short physiological time series:
        the problem of atrial fibrillation detection in implanted ventricular devices."
        American Journal of Physiology-Heart and Circulatory Physiology 300.1 (2011): H319-H325.
    The Coefficient of Sample Entropy (cosEn) is an indicator of irregularity in the input signal and hence a good indicator for AF, on short windows.
    :param segment: The input RR intervals time-series.
    :returns cosEn: The coefficient of sample entropy as presented in the paper (indicator of AF on short windows).
    """
    if len(segment) < 3:
        return 0
    r = 0.03  # initial value of the tolerance matching
    M = 2  # maximum template length

    mNc = 5  # minimum numerator count
    dr = 0.001  # tolerance matching increment  #is it ok
    A = -1000 * np.ones((M, 1))  # number of matches for m=1,...,M

    # Compute the number of matches of length M and M-1,
    # making sure that A(M) >= mNc
    compteur = 0
    while A[M - 1, 0] < mNc:
        e, A, B = comp_sampEn(segment, M, r)
        r += dr
        compteur = compteur + 1
        if compteur > 10000:
            return 0

    mRR = compute_mean(segment)
    if mRR > 0:
        cosEn = e[M - 1, 0] + np.log(2 * (r - dr)) - np.log(mRR)
    else:
        cosEn = 0
    return cosEn


def comp_AFEv(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:     The input RR intervals time-series.
    :returns AFEv:      The AFEv measure as described in the original paper.
    """

    # Compute dRR intervals series
    dRR = comp_dRR(segment)

    # Compute metrics
    OriginCount, IrrEv, PACEv = metrics(dRR)

    # Compute AFEvidence
    AFEv = IrrEv - OriginCount - 2 * PACEv

    return AFEv


def comp_IrrEv(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:     The input RR intervals time-series.
    :returns IrrEv:      The IrrEv measure as described in the original paper.
    """

    # Compute dRR intervals series
    dRR = comp_dRR(segment)

    # Compute metrics
    _, IrrEv, _ = metrics(dRR)
    return IrrEv


def comp_PACEv(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:     The input RR intervals time-series.
    :returns IrrEv:      The PACEv measure as described in the original paper.
    """

    # Compute dRR intervals series
    dRR = comp_dRR(segment)

    # Compute metrics
    _, _, PACEv = metrics(dRR)
    return PACEv


def comp_OriginCount(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:             The input RR intervals time-series.
    :returns OriginCount:       The OriginCount measure as described in the original paper.
    """

    dRR = comp_dRR(segment)
    dRR = np.vstack((dRR[1:], dRR[:-1])).transpose().astype(float)
    # COMPUTE OriginCount
    OCmask = 0.02
    os = np.sum(np.abs(dRR) <= OCmask, axis=1)
    OriginCount = np.sum(os == 2)
    return OriginCount


def comp_AVNN(segment):
    """ This function returns the mean RR interval (AVNN) over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :returns AVNN:  The mean RR interval over the segment.
    """

    return compute_mean(segment)


def comp_SDNN(segment):
    """ This function returns the standard deviation over the RR intervals (SDNN) found in the input.
    :param segment: The input RR intervals time-series.
    :returns SDNN:  The std. dev. over the RR intervals.
    """

    return compute_std(segment, ddof=1)


def comp_SEM(segment):
    """ This function returns the Standard Error of the Mean (SEM) over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :returns SEM:  The Standard Error of the Mean (SEM) over the segment.
    """
    if (len(segment)) > 1:
        return compute_std(segment, ddof=1) / np.sqrt(len(segment))
    else:
        return 0


def comp_minRR(segment):
    """ This function returns the Standard Error of the Mean (SEM) over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :returns minRR:  The Standard Error of the Mean (SEM) over the segment.
    """
    return minimum(segment)


def comp_medHR(segment):
    """ This function returns the Median Heart Rate (MedHR) over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :returns medHR:  The Median Heart Rate (medHR) over the segment.
    """
    segment = np.asarray(segment)
    segment = segment[segment > 0]
    if len(segment) > 0:
        return compute_median(60 / segment)
    else:
        return 0


def comp_PNN20(segment):
    """ This function returns the percentage of the RR interval differences above .02 over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :returns PNN20:  The percentage of the RR interval differences above .02.
    """
    if len(segment) > 0:
        return 100 * np.sum(np.abs(np.diff(segment)) > 0.02) / (len(segment))
    else:
        return 0


def comp_PNN50(segment):
    """ This function returns the percentage of the RR interval differences above .05 over a segment of RR time series.
    :param segment: The input RR intervals time-series.
    :returns PNN50:  The percentage of the RR interval differences above .05.
    """
    if len(segment) > 0:
        return 100 * np.sum(np.abs(np.diff(segment)) > 0.05) / (len(segment))
    else:
        return 0


def comp_RMSSD(segment):
    """ This function returns the RMSSD measure over a segment of RR time series.
        https://www.biopac.com/application/ecg-cardiology/advanced-feature/rmssd-for-hrv-analysis/
    :param segment: The input RR intervals time-series.
    :returns PNN20:  The RMSSD measure over the RR interval time series.
    """

    return np.sqrt(compute_mean(np.diff(segment) ** 2))


def comp_CV(segment):
    """ This function returns the Coefficient of Variation (CV) measure over a segment of RR time series.
    https://en.wikipedia.org/wiki/Coefficient_of_variation
    :param segment: The input RR intervals time-series.
    :returns CV:  The CV measure over the RR interval time series.
    """
    segment = np.asarray(segment)
    segment = segment[segment > 0]
    if len(segment) > 0:
        return compute_std(segment, ddof=1) / compute_mean(segment)
    else:
        return 0


def comp_sq_map(segment):
    """ This function implements the algorithm of:
            Zabihi, Morteza, et al.
            "Detection of atrial fibrillation in ECG hand-held devices using
            a random forest classifier."
            2017 Computing in Cardiology (CinC). IEEE, 2017.
        In particular, this functions returns the coefficients of the mapping RR[i] --> (mean(RR) - RR[i]) ** 2
    """

    X = np.hstack((segment.reshape(-1, 1), (segment ** 2).reshape(-1, 1)))
    y = (compute_mean(segment) - segment) ** 2
    reg = LinearRegression()
    reg.fit(X, y)
    return tuple(np.insert(reg.coef_, 0, reg.intercept_))


def comp_poincare(segment):
    x_old = segment[:-1]
    y_old = segment[1:]
    alpha = -np.pi / 4
    rotation_matrix = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rri_rotated = np.dot(rotation_matrix(alpha), np.array([x_old, y_old]))
    x_new, y_new = rri_rotated
    if len(x_new) > 1:
        sd1 = compute_std(y_new, ddof=1)
        sd2 = compute_std(x_new, ddof=1)
    else:
        sd1 = 0
        sd2 = 0
    return sd1, sd2


def comp_SD1(segment):
    return comp_poincare(segment)[0]


def comp_SD2(segment):
    return comp_poincare(segment)[1]


def comp_sq_map_intercept(segment):
    """ This function implements the algorithm of:
            Zabihi, Morteza, et al.
            "Detection of atrial fibrillation in ECG hand-held devices using
            a random forest classifier."
            2017 Computing in Cardiology (CinC). IEEE, 2017.
        In particular, this functions returns the intercept coefficient of the mapping RR[i] --> (mean(RR) - RR[i]) ** 2
    """

    return comp_sq_map(segment)[0]


def comp_sq_map_linear(segment):
    """ This function implements the algorithm of:
            Zabihi, Morteza, et al.
            "Detection of atrial fibrillation in ECG hand-held devices using
            a random forest classifier."
            2017 Computing in Cardiology (CinC). IEEE, 2017.
        In particular, this functions returns the linear coefficient of the mapping RR[i] --> (mean(RR) - RR[i]) ** 2
    """

    return comp_sq_map(segment)[1]


def comp_sq_map_quadratic(segment):
    """ This function implements the algorithm of:
            Zabihi, Morteza, et al.
            "Detection of atrial fibrillation in ECG hand-held devices using
            a random forest classifier."
            2017 Computing in Cardiology (CinC). IEEE, 2017.
        In particular, this functions returns the quadratic coefficient of the mapping RR[i] --> (mean(RR) - RR[i]) ** 2
    """

    return comp_sq_map(segment)[2]



