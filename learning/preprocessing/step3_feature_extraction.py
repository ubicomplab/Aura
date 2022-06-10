import pickle

from pyquaternion import Quaternion

from make_dataset import MAG_NAMES3
from utils import load_resampled_data, MAG_RAW_NAMES, save_data, load_norm_data, NUM_RX_COILS, MAG_RAW_NAMES_1, \
    MAG_RAW_NAMES_2, MAG_RAW_NAMES_3, ALL_MAG_RAW_NAMES, load_segmented_data
import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt

NUM_PCA = 9 * NUM_RX_COILS
PCA_NAMES = ['pca%d' % x for x in range(NUM_PCA)]
PCA_NAMES_1 = PCA_NAMES[0:9]
PCA_NAMES_2 = PCA_NAMES[9:18] if NUM_RX_COILS >= 2 else []
PCA_NAMES_3 = PCA_NAMES[18:27] if NUM_RX_COILS >= 3 else []

TRIAL = 'test2'  # 't3_march10'  # file to operate on
MEAN_TRIAL = None  # file to grab means from, None if should recompute
VARIANTS = [""]

CUTOFF = 0#1000
CHANNEL_1_LOW = 0.009

def row_dot(a, b):
    return np.einsum('ij,ij->i', a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def fix_signs(data):
    # plt.plot(data)
    for (i, dim) in zip(*np.where(np.diff(np.sign(data), axis=0))):
        no_switch_diff = data[i + 1, :] - data[i, :]
        switch_diff = -data[i + 1, :] - data[i, :]
        if np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff):
            data[i+1:, :] *= -1

    # plt.figure()
    # plt.plot(data)data[1,:
    # plt.show()
    return data


def fix_low_on_channel1(data):
    counter = 0
    for i in range(len(data)-1):
        if np.abs(data[i, 0]) < CHANNEL_1_LOW or np.abs(data[i+1, 0]) < CHANNEL_1_LOW:
            no_switch_diff = data[i + 1, :] - data[i, :]
            temp = np.copy(data[i + 1])
            temp[1] *= -1
            switch_diff_rx2 = temp - data[i, :]
            temp[1] *= -1
            temp[2] *= -1
            switch_diff_rx3 = temp - data[i, :]
            if (np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff_rx2)) & (np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff_rx3)):
                data[i + 1:, :] *= -1
                print("full sign flip")
            else:
                if np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff_rx2):
                    data[i + 1, 1] *= -1
                    counter += 1
                if np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff_rx3):
                    data[i + 1, 2] *= -1
                    counter += 1
    print("Recovered", counter, "points")
    return data

t = 0.011
T = np.array([[1, t, 0.5 * (t**2)],
              [0, 1,            t],
              [0, 0,            1]])
O = np.array([1,0,0])
T_COV = np.eye(3) * 1
O_COV = np.eye(3) * 1

# def fix_signs_independently(data):
#     kf = KalmanFilter(T, O, T_COV, O_COV,)
#     count = 0
#     last_mean = None
#     last_cov = None
#     for i in range(len(data) - 1):
#         kf.filter_update(
#             filtered_state_means[t],
#             filtered_state_covariances[t],
#             data.observations[t + 1],
#             transition_offset=data.transition_offsets[t],
#         )
#         no_switch_diff = data[i + 1, :] - data[i, :]
#         switch_diff = -data[i + 1, :] - data[i, :]
#         if (data[i, dim] > .02) & (np.linalg.norm(no_switch_diff) > np.linalg.norm(switch_diff)):
#             data[i+1:, dim] *= -1
#             count += 1
#     print(f"Flipped an additional {count} signs")
#     return data


def main(trial):
    for variant in VARIANTS:
        norm_data = None
        if MEAN_TRIAL is not None:
            norm_data = load_norm_data(MEAN_TRIAL, variant)
        try:
            mag_data, offs, pos, rot = load_resampled_data(trial, variant=variant)
            use_pos = True
        except OSError:
            print("Can't find resampled file, falling back to raw data only")
            mag_data = load_segmented_data(trial)
            use_pos = False
            pos = np.zeros((mag_data.shape[0], 0))
            rot = np.zeros((mag_data.shape[0], 0))

        if TRIAL == 't16':
            good_data1 = np.all(np.abs(mag_data) < 1.1, axis=1)
        else:
            good_data1 = np.all(np.abs(mag_data) < .75, axis=1)
        # good_data2 = np.all(np.abs(mag_data[:, [0,1,2]]) > 0.007, axis=1)

        b, a = scipy.signal.butter(5, 10 / (120 / 2), btype='lowpass')
        pos_filt = scipy.signal.filtfilt(b, a, pos, axis=0)
        good_data3 = np.linalg.norm(pos_filt-pos, axis=1) < 8

        # apply gain calibration
        func = pickle.load(open("interp.pkl", 'rb'))
        plt.figure()
        plt.plot(mag_data[:,6])
        noise, gain, bias = (0.1761193, 0.58924042, 0.09726018)
        # bias = np.repeat(gain * noise - offs, 3, axis=1)
        noise = np.repeat((offs + bias) / gain, 3, axis=1)
        # mag_data = np.sign(mag_data) * np.sqrt(((np.abs(mag_data) + 0.08888841 -.00059) / 1.53998297) ** 2 - 0.06149381 ** 2)
        mag_data = np.sign(mag_data) * np.sqrt(((np.abs(mag_data) + bias) / gain) ** 2 - noise ** 2)
        print(f"NaNs: {np.sum(np.isnan(mag_data))} out of {np.prod(mag_data.shape)}")
        mag_data = np.nan_to_num(mag_data)
        # mag_data = np.sign(mag_data) * (np.abs(mag_data) + func(np.abs(mag_data)))
        plt.plot(mag_data[:,6])
        # plt.show()
        tx1 = fix_signs(mag_data[:, [0, 3, 6]])
        tx2 = fix_signs(mag_data[:, [1, 4, 7]])
        tx3 = fix_signs(mag_data[:, [2, 5, 8]])

        tx1 = fix_low_on_channel1(tx1)
        tx2 = fix_low_on_channel1(tx2)
        tx3 = fix_low_on_channel1(tx3)

        # if TRIAL == 't4':
        #     tx2[63262:, :] *= -1
        # tx1 = fix_signs_independently(tx1)
        # tx2 = fix_signs_independently(tx2)
        # tx3 = fix_signs_independently(tx3)

        plt.figure()
        plt.plot(tx2)
        plt.show()

        mag_filt = scipy.signal.filtfilt(b, a, np.hstack((tx1, tx2, tx3)), axis=0)
        good_data4 = np.linalg.norm(mag_filt-np.hstack((tx1, tx2, tx3)), axis=1) < .03

        # apply channel gains
        gain = np.diag([1.1259992, 1.28982851, 1.3338626])
        tx1 = np.matmul(tx1, gain)
        tx2 = np.matmul(tx2, gain)
        tx3 = np.matmul(tx3, gain)

        def swap_axes(tx):
            # return np.matmul(np.array([[1,0,0], [0,0,1], [0,-1,0]]), tx.T).T
            return np.matmul(np.array([[1,0,0], [0,0,-1], [0,-1,0]]), tx.T).T

        tx1 = swap_axes(tx1)
        tx2 = swap_axes(tx2)
        tx3 = swap_axes(tx3)

        # b, a = scipy.signal.butter(1, 10 / (120 / 2), btype='lowpass')
        b, a = scipy.signal.butter(2, 10 / (90 / 2), btype='lowpass')
        tx1_filt = scipy.signal.filtfilt(b, a, tx1, axis=0)
        tx2_filt = scipy.signal.filtfilt(b, a, tx2, axis=0)
        tx3_filt = scipy.signal.filtfilt(b, a, tx3, axis=0)

        t_tx1 = np.linspace(1, len(tx1), len(tx1))
        t_tx2 = t_tx1 + 3 / 11
        t_tx3 = t_tx1 + 6 / 11
        plt.figure()
        plt.plot(tx1)
        plt.plot(tx1_filt)
        plt.plot(tx2)
        plt.plot(tx2_filt)
        plt.plot(tx3)
        plt.plot(tx3_filt)
        f_tx1 = scipy.interpolate.interp1d(t_tx1, tx1_filt, axis=0, assume_sorted=True, fill_value='extrapolate')
        f_tx2 = scipy.interpolate.interp1d(t_tx2, tx2_filt, axis=0, assume_sorted=True, fill_value='extrapolate')
        f_tx3 = scipy.interpolate.interp1d(t_tx3, tx3_filt, axis=0, assume_sorted=True, fill_value='extrapolate')

        tx1_filt = f_tx1(t_tx1)
        tx2_filt = f_tx2(t_tx1)
        tx3_filt = f_tx3(t_tx1)
        plt.plot(tx1_filt)
        plt.plot(tx2_filt)
        plt.plot(tx3_filt)

        good_data = good_data1 & good_data3 & good_data4  # & good_data2
        print(f"Keeping {np.sum(good_data)} out of {len(good_data)}")
        # good_data = np.all(tx1 < 0.6, axis=1) & np.all(tx2 < 0.6, axis=1) & np.all(tx3 < 0.6, axis=1)




        tx1_norot = np.zeros(tx1.shape)
        tx2_norot = np.zeros(tx1.shape)
        tx3_norot = np.zeros(tx1.shape)
        for i, _rot in enumerate(rot):
            q = Quaternion(_rot).conjugate
            # q = Quaternion([1,0,0,0])
            tx1_norot[i] = q.rotate(tx1_filt[i])
            tx2_norot[i] = q.rotate(tx2_filt[i])
            tx3_norot[i] = q.rotate(tx3_filt[i])

        plt.figure()
        # plt.plot(tx3_norot[np.abs(pos[:,0]) < 5])
        plt.plot(tx3_norot)
        plt.title("tx3_norot")
        # plt.show()
        # plt.figure()
        # plt.plot(tx1_filt)
        # plt.figure()
        # plt.plot(tx2_filt)
        # plt.figure()
        # plt.plot(tx2_filt)


        plt.figure()
        plt.plot(tx1_norot)
        plt.figure()
        plt.plot(tx2_norot)
        plt.figure()
        plt.plot(tx3_norot)

        tx1 = tx1_filt[good_data, :]
        tx2 = tx2_filt[good_data, :]
        tx3 = tx3_filt[good_data, :]
        tx1_norot = tx1_norot[good_data, :]
        tx2_norot = tx2_norot[good_data, :]
        tx3_norot = tx3_norot[good_data, :]
        pos = pos[good_data, :]
        rot = rot[good_data, :]



        # tx1 = (mag_data[:, [0, 3, 6]])
        # tx2 = (mag_data[:, [1, 4, 7]])
        # tx3 = (mag_data[:, [2, 5, 8]])


        np.savetxt("tx1.txt", tx1)
        np.savetxt("tx2.txt", tx2)
        np.savetxt("tx3.txt", tx3)


        all_norot = np.hstack((tx1_norot, tx2_norot, tx3_norot))
        np.savetxt(f"norot_{TRIAL}.txt", all_norot)
        all_rot = np.hstack((tx1, tx2, tx3))
        np.savetxt(f"rot_{TRIAL}.txt", all_rot)

        dot_products = [np.abs(row_dot(tx1, tx2)), np.abs(row_dot(tx2, tx3)), np.abs(row_dot(tx1, tx3))]
        norms = [np.linalg.norm(tx1, axis=1), np.linalg.norm(tx2, axis=1), np.linalg.norm(tx3, axis=1)]
        cross = [row_dot(tx1, np.cross(tx2, tx3))]
        features = np.vstack(dot_products + norms + cross).T

        print(np.sum(dot_products))

        # features_norm = ((features.T - np.mean(features, axis=1)) / np.std(features, axis=1)).T
        features_norm = features
        # pos = pos * 1000
        save_data(features_norm, pos, rot, trial)
        plt.show()
        # # print("WARNING: FLIPPING Q")
        # # df.qw = -df.qw
        # # df.qx = -df.qx
        # # df.qy = -df.qy
        # # df.qz = -df.qz
        # df = df.iloc[CUTOFF:]
        #
        # df.drop("is_valid", axis="columns", inplace=True)
        # for rx_coil in range(NUM_RX_COILS,3):
        #     for col in ALL_MAG_RAW_NAMES[rx_coil]:
        #         df.drop(col, axis="columns", inplace=True)
        #     for col in [f"{axis}{rx_coil+1}" for axis in "xyz"]:
        #         df.drop(col, axis="columns", inplace=True)
        #
        # # print(df)
        # save_data(df, trial, variant=variant, norm_data=norm_data)
        # df_down = df.iloc[::1, :]
        # df_sampled = df_down.sample(frac=1).reset_index(drop=True)
        # print(df_sampled.columns)
        # save_data(df_sampled, trial, variant=variant + "_shuffle", norm_data=norm_data)


# adapted from https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
def do_pca(df, columns=MAG_RAW_NAMES_1, name=PCA_NAMES_1, norm_data=None, flag=1):
    data = df[columns].as_matrix()
    if norm_data is None:
        norm_data = {'pre_pca_means':  data.mean(axis=0)}

    data -= norm_data['pre_pca_means']

    if 'pca_vecs' not in norm_data:
        print("computing pca")
        # calculate the covariance matrix
        R = np.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = scipy.linalg.eigh(R)

        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        print(evals)
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :9]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        df.reset_index(inplace=True, drop=True)
        norm_data['pca_vecs'] = evecs.T

    df[name] = pd.DataFrame(np.dot(norm_data['pca_vecs'], data.T).T)
    return norm_data


def compute_norm(data, norm_data=None):
    if norm_data is None:
        norm_data = {}
    if PCA_NAMES[0] in data.columns:
        norm_data['pca_mean'] = data[PCA_NAMES].mean()
        norm_data['pca_scale'] = data[PCA_NAMES].std()
    # print("WARNING! scale fixed to 1")
    norm_data['mag_raw_mean'] = data[MAG_RAW_NAMES].mean()
    # norm_data['mag_raw_mean'] = np.zeros([1, NUM_RX_COILS*9])
    norm_data['mag_raw_scale'] = data[MAG_RAW_NAMES].std()
    # norm_data['mag_raw_scale'] = np.ones([1, NUM_RX_COILS*9])
    norm_data['pos_mean'] = data[['x', 'y', 'z']].mean()
    # norm_data['pos_mean'] = np.array([0,0,0])
    norm_data['pos_scale'] = data[['x', 'y', 'z']].std()
    # norm_data['pos_scale'] = np.array([1,1,1])

    return norm_data


def normalize(data, norm_data=None):
    if norm_data is None or 'pos_scale' not in norm_data:
        print("computing norms")
        norm_data = compute_norm(data, norm_data)

    if PCA_NAMES[0] in data.columns:
        data[PCA_NAMES] = (data[PCA_NAMES] - norm_data['pca_mean']) / norm_data['pca_scale']
    data[MAG_RAW_NAMES] = (data[MAG_RAW_NAMES] - norm_data['mag_raw_mean']) / (norm_data['mag_raw_scale'])
    data[['x', 'y', 'z']] = (data[['x', 'y', 'z']] - norm_data['pos_mean']) / norm_data['pos_scale']

    return norm_data


if __name__ == "__main__":
    main(TRIAL)
