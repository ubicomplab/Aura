import pickle

from utils import load_segmented_data, load_extracted_data, MAG_RAW_NAMES, MAG_FILT_NAMES, save_resampled_data, \
    load_opti_data, load_extracted_opti_data, OPTI_NAMES, load_buffer_data, save_segmented_data, ROT, POS, \
    get_opti_file, load_extracted_vicon_data
from preprocessing.segmentation2 import segment, get_simple_state
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.interpolate
import pandas as pd
import seaborn as sns
from pyquaternion import Quaternion

TRIAL = "test2"

OPTI = True


def filter_in_box(filtered_data):
    x = (filtered_data.x < -.02) & (filtered_data.x > -.03)
    y = (filtered_data.y < -.50) & (filtered_data.y > -.51)
    z = (filtered_data.z < .285) & (filtered_data.z > .275)
    filtered = x & y & z
    return filtered_data[filtered]


def filter_good_data(filtered_data, drop_high=False, drop_low=False, only_best=False):

    total_points = len(filtered_data)
    if only_best:
        drop_high = True
    valid_data = filtered_data[filtered_data.is_valid]

    if drop_high:
        is_valid = np.all(valid_data[MAG_RAW_NAMES] < np.cbrt(17000), axis=1)
        valid_data = valid_data[is_valid]
    if drop_low:
        is_valid = np.partition(valid_data[MAG_RAW_NAMES], 4, axis=1)[:, 4] > np.cbrt(200)  # note using valid_data here instead of raw_data, just because it's easier
        valid_data = valid_data[is_valid]
    if only_best:
        norm = np.linalg.norm(valid_data[MAG_RAW_NAMES].values, axis=1)
        # is_valid = np.all(valid_data[MAG_RAW_NAMES] > 100, axis=1)
        is_valid = norm > np.cbrt(2000)
        valid_data = valid_data[is_valid]

    remaining_points = len(valid_data)
    print("Kept %d out of %d points. (%d%%)" % (remaining_points, total_points, remaining_points / total_points * 100))
    return valid_data


def next_power_of_two(n):
    y = np.floor(np.log2(n))
    return (int)(np.power(2, y + 1))


def trim_to_times(data, start, stop):
    start_index = np.argmin(abs(data.time - start))
    stop_index = np.argmin(abs(data.time - stop))
    print("Reducing 0:%d to %d:%d" % (len(data)-1, start_index, stop_index))
    return data.iloc[start_index:stop_index+1].reset_index()


def plot_alignment(data, pos, rot):
    pass
    # opti_distance = np.linalg.norm(pos, axis=1)
    # # mag_distance = 5/np.cbrt(np.sqrt(np.sum(raw_data_slow[MAG_RAW_NAMES]**2,axis=1)))
    # mag_distance2 = 5 / np.linalg.norm(data, axis=1)
    #
    # b, a = scipy.signal.butter(2, .1 / (217 / 2), btype='highpass')
    # b, a = scipy.signal.butter(2, .1 / (217 / 2), btype='lowpass')
    #
    # filtered_mag_diff = np.diff(scipy.signal.filtfilt(b, a, np.abs(data), axis=0), axis=0)
    # mag_sig = np.linalg.norm(filtered_mag_diff, axis=1)
    # # opti_distance = scipy.signal.filtfilt(b, a, opti_distance)
    # # mag_distance2 = scipy.signal.filtfilt(b, a, mag_distance2)
    # x = (opti_distance - np.mean(opti_distance)) / np.std(opti_distance)
    # y = (mag_distance2 - np.mean(mag_distance2)) / np.std(mag_distance2)
    #
    # # from scipy.spatial.distance import euclidean
    # #
    # # from fastdtw import fastdtw
    # #
    # # distance, path = fastdtw(x, y, dist=lambda a, b: np.sum(1 - np.sign(a) * np.sign(b)))
    # # print(distance)
    # # px, py = zip(*path)
    #
    # xs = np.array_split(x, 100)
    # ys = np.array_split(y, 100)
    # cs = []
    # for i in range(len(xs)):
    #     c = scipy.signal.correlate(xs[i], ys[i])
    #     cs.append(np.argmax(c))
    #
    # plt.figure()
    # plt.plot(cs)
    #
    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(x)
    # plt.plot(y)
    #
    # #
    # # plt.figure()
    # # plt.plot(px, py)
    # # plt.figure()
    # # plt.figure()
    # # plt.plot(x[np.array(px)])
    # # plt.plot(y[np.array(py)])


def transform_rot(mag_data, pos, rot):
    rot_quat = [Quaternion(x) for x in rot]
    # rot_quat = [(x * y.conjugate).degrees for x, y in zip(rot_quat[1:], rot_quat[:-1])]
    opti_sig = [x.elements for x in rot_quat]
    opti_sig /= np.std(opti_sig)

    mag_sig = mag_data / np.std(mag_data)

    return mag_sig, opti_sig


def transform_pos(mag_data, pos, rot):
    opti_distance = np.linalg.norm(pos, axis=1)
    mag_distance = 5 / np.linalg.norm(mag_data, axis=1)

    # b, a = scipy.signal.butter(2, .1 / (217 / 2), btype='highpass')
    # b, a = scipy.signal.butter(2, .1 / (217 / 2), btype='lowpass')
    #
    # filtered_mag_diff = np.diff(scipy.signal.filtfilt(b, a, np.abs(data), axis=0), axis=0)
    # mag_sig = np.linalg.norm(filtered_mag_diff, axis=1)
    # opti_distance = scipy.signal.filtfilt(b, a, opti_distance)
    # mag_distance2 = scipy.signal.filtfilt(b, a, mag_distance2)
    opti_sig = (opti_distance - np.mean(opti_distance)) / np.std(opti_distance)
    mag_sig = (mag_distance - np.mean(mag_distance)) / np.std(mag_distance)

    return mag_sig, opti_sig
#
# def find_shift(mag, opti):
#     plt.figure()
#     plt.plot(mag)
#     plt.plot(opti)
#
#     corr = scipy.signal.correlate(mag, opti)
#     print(np.argmax(corr))
#
#
#     plt.figure()
#     plt.plot(mag)
#     plt.plot(opti)
#     plt.show()

def resample_and_align(mag_data, pos, rot, preprocessor, transformer=transform_pos, mag_rate=90.90, opti_rate=120):

    # filter the raw file
    # b, a = scipy.signal.butter(8, 10 / (217/2))
    # b, a = scipy.signal.butter(8, 0.1)
    # data_filt = scipy.signal.filtfilt(b, a, np.cbrt(scipy.signal.filtfilt(b, a, mag_data, axis=0)), axis=0)

    # start_time = max(data_filt.time.values[0], opti_data.time.values[0])
    # stop_time = min(data_filt.time.values[-1], opti_data.time.values[-1])

    mag_data, pos, rot = preprocessor(mag_data, pos, rot)

    # mag_sig, opti_sig = transformer(mag_data, pos, rot)
    # b, a = scipy.signal.butter(5, 10 / (120 / 2), btype='lowpass')
    # rot_diff_filt = scipy.signal.filtfilt(b, a, rot_diff)
    # t_mag_est = np.linspace(0, len(mag_data), len(mag_data))
    # t_rot_est = np.linspace(0, len(mag_data), len(rot))
    # plt.figure()
    # plt.plot(t_mag_est, mag_sig)
    # # plt.figure()
    # plt.plot(t_rot_est, opti_sig)
    # # plt.show()

    f_pos_interp = scipy.interpolate.interp1d(list(range(len(pos))), pos, axis=0, assume_sorted=True, fill_value='extrapolate')
    pos_interp = f_pos_interp(np.linspace(0,len(pos)-1, len(mag_data)))
    f_rot_interp = scipy.interpolate.interp1d(list(range(len(rot))), rot, axis=0, assume_sorted=True, fill_value='extrapolate')
    rot_interp = f_rot_interp(np.linspace(0,len(pos)-1, len(mag_data)))

    plot_alignment(mag_data, pos_interp, rot_interp)

    mag_sig, opti_sig = transformer(mag_data, pos_interp, rot_interp)


    plt.figure()
    plt.plot(mag_data)
    plt.figure()
    plt.plot(mag_sig)
    plt.plot(opti_sig)
    PADDING = 50
    SKIP = 1

    def test_alignment(start_pos):
        corr = scipy.signal.correlate(mag_sig[start_pos:start_pos + 1000],
                                      opti_sig[start_pos - PADDING:start_pos + 1000 + PADDING], 'valid')
        return np.argmax(corr) - PADDING, corr


    print("Start alignment: ", test_alignment(100)[0])
    alignments = []
    all_corr = []
    for i in range(PADDING, len(mag_sig) - 1000 - PADDING * 2, SKIP):
        peak, corr = test_alignment(i)
        alignments.append(peak)
        all_corr.append(corr)
    all_corr = np.array(all_corr)
    plt.figure()
    plt.imshow(all_corr)
    plt.figure()
    plt.plot(alignments)
    print("End alignment: ", test_alignment(-1101)[0])



    b, a = scipy.signal.butter(2, 1 / (9000/SKIP / 2), btype='lowpass')

    filtered_alignments = scipy.signal.filtfilt(b, a, alignments)
    plt.plot(filtered_alignments)



    f_drift_interp = scipy.interpolate.interp1d(np.linspace(0, len(pos_interp)-1, len(filtered_alignments)), filtered_alignments, axis=0, assume_sorted=True, fill_value='extrapolate')
    drift_interp = f_drift_interp(np.linspace(0,len(pos_interp)-1, len(pos_interp)))


    sns.set(context="paper", style="white", font="Lato")
    current_palette = sns.color_palette()
    sns.palplot(current_palette)
    fig = plt.figure(figsize=(3.3, 2))
    ax = fig.add_subplot(111)
    t = np.linspace(0, len(mag_sig) / 90, len(alignments))
    ax.plot(t, alignments, label="Raw alignment", color=current_palette[0])
    ax.plot(t, filtered_alignments, label="Filtered alignment", color=current_palette[1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Alignment (samples)")
    plt.legend()
    fig.subplots_adjust(bottom=0.25, left=0.18)


    pos_interp2 = f_pos_interp(np.linspace(0, len(pos) - 1, len(mag_data)) - drift_interp * 240/95)
    rot_interp2 = f_rot_interp(np.linspace(0, len(pos) - 1, len(mag_data)) - drift_interp * 240/95)


    mag_sig, opti_sig = transformer(mag_data, pos_interp2, rot_interp2)

    plt.figure()
    plt.plot(mag_data)
    plt.figure()
    plt.plot(mag_sig)
    plt.plot(opti_sig)

# PAPER FIGURE
#     sns.set(context="paper", style="white", font="Lato")
#     fig = plt.figure(figsize=(3.3, 2))
#     ax = fig.add_subplot(111)
#     t = np.linspace(0, (20700-18000)/90, 20700-18000)
#     ax.plot(t, mag_sig[18000:20700], label="Aura")
#     ax.plot(t, opti_sig[18000:20700], label="Vicon")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Distance approximation")
#     ax.get_yaxis().set_ticks([])
#     plt.legend()
#     fig.subplots_adjust(bottom=0.25, left=0.08)

    def test_alignment2(start_pos):
        corr = scipy.signal.correlate(mag_sig[start_pos:start_pos+1000], opti_sig[start_pos-PADDING:start_pos+1000+PADDING], 'valid')
        return np.argmax(corr) - PADDING, corr

    alignments = []
    all_corr = []
    for i in range(PADDING, len(mag_sig) - 1000 - PADDING*2, SKIP):
        peak, corr = test_alignment2(i)
        alignments.append(peak)
        # if peak > 50:
        #     break
        all_corr.append(corr)
    all_corr = np.array(all_corr)
    plt.figure()
    plt.plot(alignments)

    plt.show()

    # func = pickle.load(open("interp.pkl", 'rb'))
    # mag_data = np.sign(mag_data) * (np.abs(mag_data) + func(np.abs(mag_data)))
    # plt.plot(mag_data)

    return mag_data, pos_interp2, rot_interp2

    # # data_trim = trim_to_times(data_raw, start_time, stop_time)
    # # opti_trim = trim_to_times(opti_raw, start_time, stop_time)
    # plot_alignment(data_trim, opti_trim)
    #
    # f_interp = scipy.interpolate.interp1d(data_trim.time, data_trim[MAG_RAW_NAMES].values, axis=0, assume_sorted=True,
    #                                       fill_value='extrapolate')
    #
    # resampled = f_interp(opti_trim.time)
    # # data_trim[MAG_RAW_NAMES] = data_reinterp
    #
    # plot_alignment(pd.DataFrame(resampled, columns=MAG_RAW_NAMES), opti_trim)
    #
    # # # grab the first and last points to search for in the raw file for alignment
    # # first_point = data_slow[MAG_RAW_NAMES].iloc[0]
    # # last_point = data_slow[MAG_RAW_NAMES].iloc[-1]
    # # first_point_opti = data_slow[OPTI_NAMES].iloc[0]
    # # last_point_opti = data_slow[OPTI_NAMES].iloc[-1]
    # #
    # # # find the first and last points in the raw file
    # # start = np.argmax(np.all((data_raw[MAG_RAW_NAMES] == first_point).as_matrix(), axis=1))
    # # end = np.argmax(np.all((data_raw[MAG_RAW_NAMES] == last_point).as_matrix(), axis=1))
    # # start_opti = np.argmin(np.linalg.norm((opti_raw[OPTI_NAMES] - first_point_opti).values, axis=1))
    # # end_opti = np.argmin(np.linalg.norm((opti_raw[OPTI_NAMES] - last_point_opti).values, axis=1))
    # #
    # # print(start)
    # # print(end)
    # # print(start_opti)
    # # print(end_opti)
    #
    # # # cut the raw file to match extents of the slow file
    # # data_raw_trim = data_raw[start:end+1,:]
    # # opti_raw_trim = opti_raw.values[start_opti:end_opti+1,:]
    #
    # # # len_slow = len(data_slow)
    # # len_data = len(data_trim)
    # # len_opti = len(opti_trim)
    # # print(len_data)
    # # print(len_opti)
    # # # print(len_slow)
    # # scale_factor = len_opti / len_data
    # # print((0, next_power_of_two(len_data)-len_data), (0,0))
    # # pad_len = next_power_of_two(len_data)
    # # padded = np.pad(data_trim[MAG_RAW_NAMES], ((0, pad_len-len_data), (0,0)), mode='constant')
    # # # print(padded)
    # # print(len(padded))
    # # resampled = scipy.signal.resample(padded, round(pad_len*scale_factor), window='blackman')[:len_opti,:]
    #
    # # raw_data_slow = data_slow.iloc[50:-50].reset_index()
    # full = pd.concat([pd.DataFrame(opti_trim, columns=OPTI_NAMES), pd.DataFrame(resampled, columns=MAG_RAW_NAMES),
    #                   opti_trim.is_valid], axis='columns')
    # full = full.iloc[50:-50].reset_index()  # drop a few on each side due to resample instability
    # full = full.drop(['index'], axis='columns')
    #
    # plot_alignment(full, full)
    #
    # # print(data_slow)
    # # plt.plot(data_slow['m_0'])
    #
    # standard = filter_good_data(full, drop_high=False, drop_low=False)
    # nohigh = filter_good_data(full, drop_high=True, drop_low=False)
    # nohigh_nolow = filter_good_data(full, drop_high=True, drop_low=True)
    # onlybest = filter_good_data(full, only_best=True)
    # in_box = filter_in_box(full)
    #
    # plot_alignment(nohigh, nohigh)
    # print("NOT SAVING")
    # save_resampled_data(standard, trial)
    # # save_resampled_data(nohigh, trial, variant="nohigh")
    # # save_resampled_data(nohigh_nolow, trial, variant="nohighnolow")
    # # save_resampled_data(onlybest, trial, variant="onlybest")
    # # save_resampled_data(in_box, trial, variant="inbox")
    # # plt.figure()
    # # plt.plot(padded[:,0])
    # # plt.plot(full['m_0'])
    # # plt.plot(resampled[:,0])
    # plt.show()


def preprocess_calibration(mag_data, pos, rot):
    mag_data = mag_data[15726 + 855:]
    pos = pos[:-220]
    rot = rot[:-220]
    return mag_data, pos, rot


def preprocess_alignment_t1(mag_data, pos, rot):
    start_opti = 1542754023.744472
    start_mag = 1542754012.751443

    mag_trim_samples = int((start_opti - start_mag) * (1 / 0.011))
    assert (mag_trim_samples > 0)
    mag_data = mag_data[mag_trim_samples + 60:, :]

    pos = pos[:-703]
    rot = rot[:-703]
    return mag_data, pos, rot


def preprocess_alignment_t3(mag_data, pos, rot):
    # start_opti = 1542754023.744472
    # start_mag = 1542754012.751443
    #
    # mag_trim_samples = int((start_opti - start_mag) * (1 / 0.011))
    # assert (mag_trim_samples > 0)
    mag_data = mag_data[420:-580, :]
    pos = pos[0:]
    rot = rot[0:]
    #
    # pos = pos[:-703]
    # rot = rot[:-703]
    return mag_data, pos, rot


def preprocess_alignment_t4(mag_data, pos, rot):
    # start_opti = 1542754023.744472
    # start_mag = 1542754012.751443
    #
    # mag_trim_samples = int((start_opti - start_mag) * (1 / 0.011))
    # assert (mag_trim_samples > 0)
    mag_data = mag_data[10000+7:-6, :]
    pos = pos[24300:-850]
    rot = rot[24300:-850]
    #
    # pos = pos[:-703]
    # rot = rot[:-703]
    return mag_data, pos, rot


def preprocess_alignment_t5(mag_data, pos, rot):
    # start_opti = 1542754023.744472
    # start_mag = 1542754012.751443
    #
    # mag_trim_samples = int((start_opti - start_mag) * (1 / 0.011))
    # assert (mag_trim_samples > 0)
    mag_data = mag_data[18119+107+17:-3000+290+41+3, :]
    # pos = pos
    # rot = rot
    #
    # pos = pos[:-703]
    # rot = rot[:-703]
    return mag_data, pos, rot


def preprocess_alignment_t5a(mag_data, pos, rot):
    # start_opti = 1542754023.744472
    # start_mag = 1542754012.751443
    #
    # mag_trim_samples = int((start_opti - start_mag) * (1 / 0.011))
    # assert (mag_trim_samples > 0)
    mag_data = mag_data[28119+107+17-33-2:48200-27, :]
    pos = pos[25000:75000]
    rot = rot
    #
    # pos = pos[:-703]
    # rot = rot[:-703]
    return mag_data, pos, rot


def preprocess_alignment_direction_test3(mag_data, pos, rot):
    return mag_data[229+80:-300-30-5], pos, rot


def preprocess_alignment_direction_test4(mag_data, pos, rot):
    return mag_data[290+174+42+17+8:-400-140-32-7], pos, rot


def preprocess_alignment(mag_data, pos, rot):
    if TRIAL == "t7":
        return mag_data[484-85-4:], pos[:-2000+7], rot[:-2000+7]
    elif TRIAL == "t8":
        return mag_data[532+40+8:-150-17], pos, rot
    elif TRIAL == "t9":
        return mag_data[429+15+375+29:-318+2-375-27], pos[1000:-1000], rot[1000:-1000]
    elif TRIAL == "t10":
        return mag_data[700+24+2:-771-7-1], pos[:-1000], rot[:-1000]
    elif TRIAL == "t11":
        return mag_data[656+28-170:-400-23], pos, rot
    elif TRIAL == "t12":
        return mag_data[1160:-318-5], pos[2000:], rot[2000:]
    elif TRIAL == "t13":
        return mag_data[310+52:-397-14], pos[:], rot[:]
    elif TRIAL == "t14":
        return mag_data[2850+92-19:-330-14], pos[6000:], rot[6000:]
    elif TRIAL == "t15":
        return mag_data[1140:-460], pos[1700:], rot[1700:]
    elif TRIAL == "t16":
        return mag_data[1350:17080], pos[2600:42000], rot[2600:42000]
        # return mag_data[17050:-330], pos[42000:], rot[42000:]
    elif TRIAL == "test2":
        return mag_data[330:], pos[:-533], rot[:-533]

    print("No alignment set")
    return mag_data, pos, rot



def main(trial):
    data = load_segmented_data(trial)  # loads the segmented  magnetic file
    if OPTI:
        tracking_raw = load_extracted_opti_data(trial)  # loads the raw high speed opti file
    else:
        tracking_raw = load_extracted_vicon_data(trial)  # loads the raw high speed opti file
    data = data[:len(data)] # shorten just for testing
    tracking_raw = tracking_raw.iloc[:len(tracking_raw)]  # shorten just for testing
    mag_data, pos_interp, rot_interp = resample_and_align(data, tracking_raw[POS], tracking_raw[ROT], preprocess_alignment)

    print("Saving data...")
    save_resampled_data(mag_data, pos_interp, rot_interp, trial)


if __name__ == "__main__":
    main(TRIAL)
