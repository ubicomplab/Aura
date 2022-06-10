import scipy.signal

from preprocessing.step3_feature_extraction import PCA_NAMES
from utils import load_data, load_predictions, MAG_FILT_NAMES, load_resampled_data, MAG_RAW_NAMES, load_norm_data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# v11a is good
MODEL = 'testMulti'
DATASET = 'sim'
MODEL_DATASET = 'sim'
VARIANT = '2rx'  # 'nohighnolow'#nohighnolow'
TRIAL = MODEL + "_" + MODEL_DATASET + "_" + VARIANT + "_shuffle"

CUTOFF = 0

def main():

    df = load_data(DATASET, VARIANT).iloc[CUTOFF:]
    norm_data = load_norm_data(DATASET, VARIANT)
    preds = load_predictions(TRIAL, DATASET)[CUTOFF:]
    for i, col in enumerate(['px', 'py', 'pz']):
        df[col] = preds[:,i]

    # validation_set = [i for i in range(len(df)) if (i % 10000) >= 7000]
    try:
        scale = np.squeeze(norm_data['pos_scale'].values)
    except:
        scale = norm_data['pos_scale']
    pos = np.multiply(df[['x1','y1','z1']].values, scale)
    pos_hat = np.multiply(df[['px','py','pz']].values, scale)

    error = (pos - pos_hat) * 1000
    error_mag = np.linalg.norm(error, axis=1)

    print(np.mean(error_mag))
    print(np.median(error_mag))
    print(np.sqrt(np.mean(error_mag**2)))

    # print(np.mean(error_mag[validation_set]))
    # print(np.median(error_mag[validation_set]))
    # print(np.sqrt(np.mean(error_mag[validation_set]**2)))

    print(np.cov(error.T))

    # pos_hat_filt = scipy.signal.savgol_filter(pos_hat, 99, 2, axis=0)
    b, a = scipy.signal.butter(8, 2.2 / (230 / 2))
    pos_hat_filt = scipy.signal.filtfilt(b, a, pos_hat, axis=0)

    error_filt = (pos - pos_hat_filt) * 1000
    print(np.mean(np.abs(error_filt), axis=0))
    print(np.median(np.abs(error_filt), axis=0))
    print(np.sqrt(np.mean(error_filt**2, axis=0)))
    error_mag_filt = np.linalg.norm(error_filt, axis=1)

    print(np.mean(error_mag_filt))
    print(np.median(error_mag_filt))
    print(np.sqrt(np.mean(error_mag_filt**2)))

    print(np.argsort(error_mag_filt)[::-1])

    plt.figure()
    sns.distplot(error_mag)
    sns.distplot(error_mag_filt)

    # plt.figure()
    # sns.distplot(error_mag, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
    # sns.distplot(error_mag_filt, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))

    # plt.figure()
    # plt.plot(df[PCA_NAMES])

    for i in range(3):
        plt.figure()
        plt.plot(pos[:,i] * 1000)
        plt.plot(pos_hat[:,i] * 1000)
        plt.plot(pos_hat_filt[:,i] * 1000)

    plt.figure()
    df2 = load_resampled_data(DATASET, variant=VARIANT)
    plt.plot(df2[MAG_RAW_NAMES])
    plt.figure()
    plt.plot(df[PCA_NAMES])
    dt = .1
    v = dt
    a = .5 * (dt**2)
    trans = np.array([[1, 0, 0, v, 0, 0, a, 0, 0],
                      [0, 1, 0, 0, v, 0, 0, a, 0],
                      [0, 0, 1, 0, 0, v, 0, 0, a],
                      [0, 0, 0, 1, 0, 0, v, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, v, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, v],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    obs = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0]])
    t_cov = 1*np.eye(9) + .5*np.ones(9)
    # o_cov = np.array([[150,0,0],
    #                  [0,200,0],
    #                  [0,0,500]])+ 100*np.ones(3)#10*np.eye(3)
    # o_cov = np.array([[34.17598201, -2.5295378, -0.50223841],
    #                  [-2.5295378, 41.44479497, 20.30671876],
    #                  [-0.50223841, 20.30671876, 72.63576359]])
    # kf = KalmanFilter(n_dim_obs=3, n_dim_state=9, # position is 3-dimensional, (x,v,a) is 9-dimensional
    #               transition_matrices=trans,
    #               observation_matrices=obs,
    #               observation_covariance=o_cov)#,
    #                 # transition_covariance=t_cov,
    #                 #   )
    # print('em')
    # kf = kf.em(pos_hat[:10000,:], n_iter=3, em_vars=['transition_covariance', 'initial_state_mean', 'initial_state_covariance'])
    # print('smooth')
    # TRIM = -1#10000
    # (smoothed_state_means, smoothed_state_covariances) = kf.smooth(pos_hat[:TRIM,:])
    #
    # error_filt = (pos[:TRIM,:] - smoothed_state_means[:,:3]) * 1000
    # error_mag_filt = np.linalg.norm(error_filt, axis=1)
    #
    # print(np.mean(error_mag_filt))
    # print(np.median(error_mag_filt))
    # print(np.sqrt(np.mean(error_mag_filt ** 2)))
    #
    # for i in range(3):
    #     plt.figure()
    #     plt.plot(pos[:,i] * 1000)
    #     plt.plot(pos_hat[:,i] * 1000)
    #     plt.plot(smoothed_state_means[:,i] * 1000)

    plt.show(block=True)

if __name__ == "__main__":
    main()
