import scipy
import scipy.signal
from matplotlib import pyplot as plt
from utils import load_file_by_key, MAG_RAW_NAMES, MAG_FILT_NAMES, HAND_OPTI, save_data, HEAD_OPTI, save_extra_data, S
import numpy as np
import pandas as pd
import sys




TRIAL = 't1_march5'

USE_FILT = True
PLOT = True
SMOOTH = True
PCA = True

if USE_FILT:
    MAG = MAG_FILT_NAMES
else:
    MAG = MAG_RAW_NAMES

MAG_NAMES3 = [x+'3' for x in MAG]



def smooth_data(data):
    data[MAG] = scipy.signal.savgol_filter(data[MAG], 19, 3, axis=0)
    data[MAG_NAMES3] = scipy.signal.savgol_filter(data[MAG_NAMES3], 19, 3, axis=0)
    #data[MAG_NAMES] = scipy.signal.medfilt(data[MAG_NAMES], 5)
    # med_errors = np.linalg.norm(data[MAG_NAMES3] - scipy.signal.medfilt(data[MAG_NAMES3], 5), axis=1)
    # print(np.argsort(med_errors)[::-1])
    return data


def smooth_pca(data):
    data[PCA_NAMES] = scipy.signal.savgol_filter(data[PCA_NAMES], 19, 3, axis=0)
    return data








def main():
    df_norm = extract_coordinates(TRIAL)

    if PLOT:
        plt.figure()
        plt.plot(np.array(df_norm[MAG_NAMES3]))

    if SMOOTH:
        smooth_data(df_norm)

    if PCA:
        do_pca(df_norm)

    normalize(df_norm)

    save_data(df_norm, TRIAL)

    if PLOT:
        plt.figure()
        plt.plot(np.array(df_norm[PCA_NAMES]))
        # plt.plot(valid_data[HEAD_OPTI])
        # plt.figure()
        # plt.plot(valid_data[HAND_OPTI])
        plt.figure()
        plt.plot(np.array(df_norm[PCA_NAMES[0]]))
        plt.plot(np.array(df_norm[['x','y','z']]))
        plt.figure()
        plt.plot(np.array(df_norm[['qw','qx','qy','qz']]))
        # plt.figure()
        # plt.plot(valid_data[MAG_FILT_NAMES])
        plt.show()


if __name__ == "__main__":
    main()
