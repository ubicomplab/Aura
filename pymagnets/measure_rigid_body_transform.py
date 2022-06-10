import numpy as np

CALIBRATION_FILE = r'D:\mag_track\rigid_body_calibration\hand\markers_18_12_05_16_09_54.txt'
NUM_MARKERS = 5
def main():
    data = np.loadtxt(CALIBRATION_FILE, delimiter=',')
    num_elements = np.count_nonzero(data, axis=1)

    good_elements = data[num_elements == 15,:15]
    print(data)

if __name__ == "__main__":
    main()