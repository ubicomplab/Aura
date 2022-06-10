from utils import load_segmented_data, load_extracted_vicon_data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TRIAL = 't2'


def main(trial):
    tracking_raw = load_extracted_vicon_data(trial)  # loads the raw high speed opti file
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tracking_raw.x, tracking_raw.y, tracking_raw.z, alpha=.5, marker='.')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    print(tracking_raw)



if __name__ == "__main__":
    main(TRIAL)