import scipy.signal

from preprocessing.step3_feature_extraction import PCA_NAMES
from utils import load_data, load_predictions, MAG_FILT_NAMES, load_resampled_data, MAG_RAW_NAMES, load_norm_data, \
    MAG_D_NAMES
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# v11a is good
MODEL = 'forwarddeep'
DATASET = 'sim'
MODEL_DATASET = 'sim'
VARIANT = 'rot2'  # 'nohighnolow'#nohighnolow'
TRIAL = MODEL + "_" + MODEL_DATASET + "_" + VARIANT + "_shuffle"

CUTOFF = 0

def main():

    df = load_data(DATASET, VARIANT).iloc[CUTOFF:]
    norm_data = load_norm_data(DATASET, VARIANT)
    preds = load_predictions(TRIAL, DATASET)[CUTOFF:]

    print(preds.shape)
    m_dipole = df[MAG_D_NAMES]
    m_actual = df[MAG_RAW_NAMES]

    m_dipole_hat = m_actual + preds

    for i in range(9):
        plt.figure()
        plt.scatter(m_dipole.values[:,i], m_actual.values[:,i], s=1, c='b')
        plt.scatter(m_dipole.values[:,i], m_dipole_hat.values[:,i], s=1, c='r')
        plt.figure()
        plt.scatter(m_dipole.values[:,i], m_dipole.values[:,i] - m_dipole_hat.values[:,i], s=1, c='r')
        plt.savefig("correlation"+str(i))
        # plt.show(block=True)
    plt.show(block=True)


if __name__ == "__main__":
    main()
