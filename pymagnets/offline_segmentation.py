import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from segmentation2 import segment, get_simple_state


PLOT = True
SAVE = False
FOLDER = "test"

def get_files():
    return glob.glob(os.path.join("debug_recordings", FOLDER, "*.txt"))


def load_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            row = eval(line)
            for sample in row:
                data.append(sample)
    return np.array(data)


def process_file(file):
    data = load_data(file)

    # data = data[0:20000]


    states = []
    offs = []

    segmented = []
    segmented_x = []
    for row in range(data.shape[0]):
        if row % 1000 == 0:
            print(row / data.shape[0])
        processed, state, off = segment(data[row])
        if PLOT:
            states.append(state)
            offs.append(off)
        if processed is not None:
            segmented_x.append(row)
            segmented.append(processed.flatten())

    segmented = np.array(segmented)

    if PLOT:

        current_palette = sns.hls_palette(4)
        sns.palplot(current_palette)
        plt.figure()
        plt.plot(offs)

        fig, ax = plt.subplots()

        states = np.array([get_simple_state(s) for s in states])

        for i in range(data.shape[1]):
            ax.scatter(list(range(len(data))), data[:,i], c=np.array(current_palette)[states], s=5)

        ax.plot(data, 'black', linewidth=.5)
        #
        # PLOT_START = 0
        # PLOT_STOP = 1000
        #
        # ax.fill_between(list(range(PLOT_START, PLOT_STOP)), 0, data[PLOT_START:PLOT_STOP], where=(states[PLOT_START:PLOT_STOP]==1), facecolor='green', interpolate=False)
        # plt.twinx()
        # plt.plot(states, 'r')

        plt.figure()
        plt.plot(segmented_x, segmented)

        plt.show()

    if SAVE:
        np.save(file.replace(".txt", "-processed.npy"), segmented)


def main():
    files = get_files()#[::-1]
    for file in files:
        process_file(file)

if __name__ == "__main__":
    main()