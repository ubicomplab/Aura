import sys
from utils import load_buffer_data, save_segmented_data, progress, load_native_data, save_mean_segmented_data
from preprocessing.segmentation2 import segment, get_simple_state
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TRIAL = "test2"

NATIVE_DATA = True


def segment_data(data, show_plot=False):
    states = []
    offs = []

    segmented = []
    segmented_x = []
    for row in range(data.shape[0]):
        if row % 20000 == 0:
            progress(row / data.shape[0])
        processed, state, off = segment(data[row])
        if show_plot:
            states.append(state)
        if processed is not None:
            segmented_x.append(row)
            offs.append(off)
            segmented.append(processed.flatten())
    print()
    segmented = np.array(segmented)

    states = np.array([get_simple_state(s) for s in states])
    # diffs = np.diff((states == 0) * 1)
    # p, = np.where(diffs > 0)
    # n, = np.where(diffs < 0)
    # plt.figure()
    # plt.plot(p[1:] - n[:-1])
    # bad_indices = p[1:] - n[:-1] >
    offs = np.squeeze(offs)

    segmented = np.hstack((segmented, offs))
    if show_plot:

        current_palette = sns.hls_palette(4)
        sns.palplot(current_palette)
        plt.figure()
        plt.plot(offs)

        fig, ax = plt.subplots()

        for i in range(data.shape[1]):
            ax.scatter(list(range(len(data))), np.abs(data[:, i]), c=np.array(current_palette)[states], s=5)

        ax.plot(np.abs(data), 'black', linewidth=.5)

    plt.figure()
    plt.plot(segmented_x, segmented)
    plt.figure()
    plt.plot(np.diff(segmented_x))

    plt.figure()
    plt.plot(segmented_x, segmented)
    # 5 / np.linalg.norm(mag_data, axis=1)

    return segmented, segmented_x


def verify_frame_numbers(frame_numbers):
    diffs = np.diff(frame_numbers, axis=0)
    red_flags = ~np.isin(diffs, [1, -65535])
    assert(np.sum(red_flags) == 0)
    print("No missing data!")


def interference(data):
    size = len(data)
    on = data[int(3 * size / 4):-500]
    off = data[100:int(1 * size / 4)]
    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(on)
    # plt.figure()
    # plt.plot(off)
    # plt.show()
    mean_off = np.median(off, axis=0)
    mean_on = np.median(on, axis=0)
    percent = 0
    for i in range(9):
        percent += np.abs((mean_off[i]-mean_on[i])/mean_off[i])
    return percent/9


def main(trial):
    if NATIVE_DATA:
        print("Loading native data")
        data_raw, frame_numbers = load_native_data(trial)  # loads the raw high speed magnetic file
    else:
        print("Loading buffered data")
        data_raw = load_buffer_data(trial)  # loads the python version of high speed magnetic file
        frame_numbers = None
    # data_raw = data_raw[:10000]
    # plt.plot(data_raw)
    # plt.show()
    verify_frame_numbers(frame_numbers)

    print("Segmenting data")
    segmented, segmented_x = segment_data(data_raw, show_plot=False)

    print("Saving data")
    save_segmented_data(segmented, TRIAL)

    # interference analysis
    # percent = interference(segmented[:, 0:9])p    
    # print("Mean percent difference for", trial, "is:", 100*percent)


    # plt.show()



if __name__ == "__main__":
    main(TRIAL)
    # for i in np.arange(2, 9, 1):
    # for i in [2,3,4,5,6,7,8,10,11,12 ,13,14,15, 22,23,24,25,26,32,33,34,35,36]:
    # for i in [22, 23, 24, 25, 26, 32, 33, 34, 35]:

    #     main("t"+str(i))
    # main(TRIAL)
    plt.show()
