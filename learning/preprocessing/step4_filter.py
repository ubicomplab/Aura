import pickle

from pykalman import KalmanFilter
from pyquaternion import Quaternion

from make_dataset import MAG_NAMES3
from utils import load_resampled_data, MAG_RAW_NAMES, save_data, load_norm_data, NUM_RX_COILS, MAG_RAW_NAMES_1, \
    MAG_RAW_NAMES_2, MAG_RAW_NAMES_3, ALL_MAG_RAW_NAMES, load_segmented_data, load_results
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

NUM_PCA = 9 * NUM_RX_COILS
PCA_NAMES = ['pca%d' % x for x in range(NUM_PCA)]
PCA_NAMES_1 = PCA_NAMES[0:9]
PCA_NAMES_2 = PCA_NAMES[9:18] if NUM_RX_COILS >= 2 else []
PCA_NAMES_3 = PCA_NAMES[18:27] if NUM_RX_COILS >= 3 else []

TRIAL = 't15'  # 't3_march10'  # file to operate on
MEAN_TRIAL = None  # file to grab means from, None if should recompute
VARIANTS = [""]

CUTOFF = 0#1000
CHANNEL_1_LOW = 0.009
SKIP_KALMAN = False


def main(trial):

    pos_test, pos_test_pred = load_results(trial, variant="")

    dt = .1
    v = dt
    a = .5 * (dt**2)

    pos_test_pred_filt = np.zeros(pos_test_pred.shape)

    if TRIAL == 't15':
        covs = [30, 80, 50]
    elif TRIAL == 't16':
        covs = [20, 40, 30]
    for axis in range(3):
        trans = np.array([[1, v, a,],
                          [0, 1, v],
                          [0, 0, 1]])
        obs = np.array([[1, 0, 0]])
        t_cov = 1 * np.eye(3)
        o_cov = covs[axis] * np.eye(1)
        kf = KalmanFilter(n_dim_obs=1, n_dim_state=3, # position is 3-dimensional, (x,v,a) is 9-dimensional
                          transition_matrices=trans,
                          observation_matrices=obs,
                          observation_covariance=o_cov,
                          transition_covariance=t_cov)

        # kf = kf.em(pos_test_pred[:,axis], n_iter=3, em_vars=['transition_covariance', 'initial_state_mean', 'initial_state_covariance'])
        # kf = kf.em(pos_test[:,axis], n_iter=3, em_vars=['transition_covariance', 'initial_state_mean', 'initial_state_covariance'])
        if SKIP_KALMAN:
            pos_test_pred_filt[:, axis] = pos_test_pred[:, axis]
        else:
            print('em')
            kf = kf.em(pos_test[:1000,axis], n_iter=3, em_vars=['initial_state_mean', 'initial_state_covariance'])
            print('smooth')
            (smoothed_state_means, smoothed_state_covariances) = kf.smooth(pos_test_pred[:,axis])
            pos_test_pred_filt[:, axis] = smoothed_state_means[:,0]

        error_pre = pos_test_pred[:, axis] - pos_test[:, axis]
        error_post = pos_test_pred_filt[:, axis] - pos_test[:, axis]
        print(np.mean(np.abs(error_pre)))
        print(np.mean(np.abs(error_post)))

        all = np.hstack((pos_test, pos_test_pred, pos_test_pred_filt))
        np.savetxt("pos.csv", all)

        # plt.figure()
        # plt.plot(pos_test[:, axis])
        # plt.plot(pos_test_pred[:, axis])
        # plt.plot(pos_test_pred_filt[:, axis])
        # plt.show()

    error_pre = pos_test_pred - pos_test
    error_post = pos_test_pred_filt - pos_test

    error_pre_mag = np.linalg.norm(error_pre, axis=1)
    error_post_mag = np.linalg.norm(error_post, axis=1)
    outliers = (error_pre_mag > 100) | (error_post_mag > 100)

    error_pre_mag = error_pre_mag[~outliers]
    error_post_mag = error_post_mag[~outliers]

    print(np.mean(error_pre_mag))
    print(np.mean(error_post_mag))
    print(np.median(error_pre_mag))
    print(np.median(error_post_mag))
    print(np.sqrt(np.mean(error_pre_mag**2)))
    print(np.sqrt(np.mean(error_post_mag**2)))

    # plt.figure()
    # sns.kdeplot(error_pre_mag, cumulative=True)
    # sns.kdeplot(error_post_mag, cumulative=True)
    #
    # plt.figure()
    # sns.kdeplot(np.abs(error_post[:,0]), cumulative=True)
    # sns.kdeplot(np.abs(error_post[:,1]), cumulative=True)
    # sns.kdeplot(np.abs(error_post[:,2]), cumulative=True)

    # plt.figure()
    # plt.plot(pos_test)
    # plt.plot(pos_test_pred)
    # plt.plot(pos_test_pred_filt)
    sns.set(context="paper", style="white", font="Lato")
    if TRIAL == 't15':
        fig = plt.figure(figsize=(7,4))
        x_lim = (1890, 4590)
        t = np.linspace(0, (x_lim[1] - x_lim[0]) / 90, x_lim[1] - x_lim[0])
        axes = ['X', 'Y', 'Z']
        p = sns.color_palette()
        for axis in range(3):
            ax = fig.add_subplot(3, 1, axis+1)
            h_gt, = ax.plot(t, pos_test[x_lim[0]:x_lim[1], axis], color=p[0])
            h_raw, = ax.plot(t, pos_test_pred[x_lim[0]:x_lim[1], axis], color=p[1], alpha=.5)
            h_filt, = ax.plot(t, pos_test_pred_filt[x_lim[0]:x_lim[1], axis], color=p[1])
            if axis < 2:
                # ax.set_axis_off()
                # ax.get_xaxis().set_visible(False)
                ax.get_xaxis().set_ticks([])
            else:
                ax.set_xlabel("Time (s)")
            sns.despine()
            ax.set_ylabel(f"{axes[axis]}  (mm)")
            ax.yaxis.set_label_coords(-.08, .5)
        plt.figlegend((h_gt, h_raw, h_filt), ('Ground Truth', 'Raw Estimates', 'Filtered Estimates'))
        fig.subplots_adjust(bottom=0.25, left=0.18)
        # plt.tight_layout()

        sns.set(context="paper", style="whitegrid", font="Lato")
        fig = plt.figure(figsize=(3.33, 2))
        sns.kdeplot(error_pre_mag, cumulative=True, lw=3, label='Raw Estimates')
        sns.kdeplot(error_post_mag, cumulative=True, lw=3, label='Filtered Estimates')
        sns.kdeplot(np.abs(error_post[:,0]), cumulative=True, lw=1, label='Filtered X')
        sns.kdeplot(np.abs(error_post[:,1]), cumulative=True, lw=1, label='Filtered Y')
        sns.kdeplot(np.abs(error_post[:,2]), cumulative=True, lw=1, label='Filtered Z')
        plt.xlabel("Error (mm)")
        plt.ylabel("CDF")
        plt.xlim((0, 25))
        plt.ylim((0, 1))
        plt.legend()
        fig.subplots_adjust(bottom=0.25, left=0.18)
    else:
        fig = plt.figure(figsize=(3.33, 2.5))
        p = sns.color_palette()
        ax = fig.add_subplot(111)
        x_lim = (940, 1440)
        h_gt,   = ax.plot(pos_test[x_lim[0]:x_lim[1], 0], pos_test[x_lim[0]:x_lim[1], 2], color=p[0])
        h_raw,  = ax.plot(pos_test_pred[x_lim[0]:x_lim[1], 0], pos_test_pred[x_lim[0]:x_lim[1], 2], color=p[1], alpha=.5)
        h_filt, = ax.plot(pos_test_pred_filt[x_lim[0]:x_lim[1], 0], pos_test_pred_filt[x_lim[0]:x_lim[1], 2], color=p[1], ls='--')
        ax.set_ylabel("Z (mm)")
        ax.set_xlabel("X (mm)")
        ax.set_aspect('equal')

        plt.figlegend((h_gt, h_raw, h_filt), ('Ground Truth', 'Raw Estimates', 'Filtered Estimates'))
        fig.subplots_adjust(bottom=0.15, left=.17)

        sns.set(context="paper", style="whitegrid", font="Lato")
        fig = plt.figure(figsize=(3.33, 2))
        sns.kdeplot(error_pre_mag, cumulative=True, lw=3, label='Raw Estimates')
        sns.kdeplot(error_post_mag, cumulative=True, lw=3, label='Filtered Estimates')
        plt.xlabel("Error (mm)")
        plt.ylabel("CDF")
        plt.xlim((0, 10))
        plt.ylim((0, 1))
        plt.legend()
        fig.subplots_adjust(bottom=0.25, left=0.18)

    plt.show()


if __name__ == "__main__":
    main(TRIAL)
