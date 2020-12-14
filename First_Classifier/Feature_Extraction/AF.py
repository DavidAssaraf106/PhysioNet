import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from Preprocessing_features_Submission import compute_mean, compute_std, compute_median, minimum, maximum


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


def bsqi(refqrs, testqrs, agw=0.05, fs=500):
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
    try:
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
    except:
        F1 = 0
    else:
        F1 = 0
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
        bin_c = \
            sio.loadmat(str("C:\\Users\\David\\PhysioNet_Code\\First_Classifier\\Utils\\edges_hist.mat"))['edges'][0][
                0][
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
    if len(segment) > 2:
        # Compute dRR intervals series
        dRR = comp_dRR(segment)
        # Compute metrics
        OriginCount, IrrEv, PACEv = metrics(dRR)
        # Compute AFEvidence
        AFEV = IrrEv - OriginCount - 2 * PACEv
        return AFEV
    else:
        return 0


def comp_IrrEv(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:     The input RR intervals time-series.
    :returns IrrEv:      The IrrEv measure as described in the original paper.
    """
    if len(segment) > 0:
        # Compute dRR intervals series
        dRR = comp_dRR(segment)

        # Compute metrics
        _, IrrEv, _ = metrics(dRR)
        return IrrEv
    else:
        return 0


def comp_PACEv(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:     The input RR intervals time-series.
    :returns IrrEv:      The PACEv measure as described in the original paper.
    """
    if len(segment) > 0:
        # Compute dRR intervals series
        dRR = comp_dRR(segment)

        # Compute metrics
        _, _, PACEv = metrics(dRR)
        return PACEv
    else:
        return 0


def comp_OriginCount(segment):
    """
    This function implements the algorithm of:
        Sarkar, Shantanu, David Ritscher, and Rahul Mehra.
        "A detector for a chronic implantable atrial tachyarrhythmia monitor."
        IEEE Transactions on Biomedical Engineering 55.3 (2008): 1219-1224.
    :param segment:             The input RR intervals time-series.
    :returns OriginCount:       The OriginCount measure as described in the original paper.
    """
    if len(segment) > 2:
        dRR = comp_dRR(segment)
        dRR = np.vstack((dRR[1:], dRR[:-1])).transpose().astype(float)
        # COMPUTE OriginCount
        OCmask = 0.02
        os = np.sum(np.abs(dRR) <= OCmask, axis=1)
        OriginCount = np.sum(os == 2)
        return OriginCount
    else:
        return 0


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
    if len(segment) > 0:
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
    else:
        return 0, 0


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


def comp_diff(R_points):  # @Jeremy, for PVC detection
    R_points = np.asarray(R_points)
    cnt_diff_ecg = []
    for idx_q in range(1, len(R_points)):
        cnt_diff = R_points[idx_q] - R_points[idx_q - 1]
        cnt_diff_ecg.append(cnt_diff)
    return cnt_diff_ecg


def extraction_feature_AF(ecg, freq, features_dict, lead):
    R_points = features_dict['R']
    RR_index = R_points[:-1] * R_points[1:] > 0
    rr = np.asarray(comp_diff(R_points))
    rr = rr[RR_index]
    feat = pd.DataFrame()
    if lead == 0:
        SD1, SD2 = comp_poincare(rr)
        feat = pd.DataFrame({'cosEn_' + str(lead): [comp_cosEn(rr)],
                             'SD1_' + str(lead): [SD1],
                             'AFEv_' + str(lead): [comp_AFEv(rr)],
                             'OriginCount_' + str(lead): [comp_OriginCount(rr)],
                             'PACEV_' + str(lead): [comp_PACEv(rr)]})
    if lead == 1:
        feat = pd.DataFrame({'IrrEv_' + str(lead): [comp_IrrEv(rr)],
                             'minRR_' + str(lead): [comp_minRR(rr)]})
    if lead == 2:
        feat = pd.DataFrame({'medHR_' + str(lead): [comp_medHR(rr)],
                             'AVNN_' + str(lead): [comp_AVNN(rr)],
                             'SDNN_' + str(lead): [comp_SDNN(rr)],
                             'SEM_' + str(lead): [comp_SEM(rr)],
                             'RMSSD_' + str(lead): [comp_RMSSD(rr)]})
    if lead == 3:
        SD1, SD2 = comp_poincare(rr)
        feat = pd.DataFrame({'IrrEv_' + str(lead): [comp_IrrEv(rr)],
                             'SD2_' + str(lead): [SD2]})
    if lead == 5:
        feat = pd.DataFrame({'PNN20_' + str(lead): [comp_PNN20(rr)]})
    if lead == 8:
        feat = pd.DataFrame({'CV_' + str(lead): [comp_CV(rr)],
                             'minRR_' + str(lead): [comp_minRR(rr)]})
    if lead == 10:
        SD1, SD2 = comp_poincare(rr)
        feat = pd.DataFrame({'minRR_' + str(lead): [comp_minRR(rr)],
                             'SD1_' + str(lead): [SD1]})
    if lead == 11:
        feat = pd.DataFrame({'PNN50_' + str(lead): [comp_PNN50(rr)]})
    return feat
