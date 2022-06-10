import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

F1 = r'D:\mag_track\recordings\mag_helmholtz_2_18_11_20_14_39_18.txt'
F2 = r'D:\mag_track\recordings\natnet_helmholtz_2_18_11_20_14_39_18.txt'

ONLY_HAND = True

def extract_times(file, match=None):
    times = []
    with open(file, 'rb') as f:
        for line in f:
            if match is not None:
                if match not in line:
                    continue
            time_split = line.find(b',')
            time = eval(line[:time_split].decode('utf-8'))
            times.append(float(time))
    return np.array(times)


def main():
    t1 = extract_times(F1, None)
    t2 = extract_times(F2, b"mag_controller_v9" if ONLY_HAND else None)

    t1_diff = np.diff(t1)
    t2_diff = np.diff(t2)

    print(np.mean(t1_diff))
    print(np.mean(t2_diff))

    plt.plot(t1[10:] - t1[:-10])
    plt.plot(t2[10:] - t2[:-10])
    # plt.plot(scipy.signal.savgol_filter(t1_diff, 561, 1))
    # plt.plot(scipy.signal.savgol_filter(t2_diff, 561, 1))
    # plt.plot(scipy.signal.medfilt(t1_diff, 11))
    # plt.plot(scipy.signal.medfilt(t2_diff, 11))
    # plt.plot(t1)
    # plt.plot(t2)
    # plt.show()

    plt.figure()
    plt.plot(t1)
    plt.plot(t2)
    plt.figure()
    plt.plot(np.diff(t1))
    plt.plot(np.diff(t2))
    plt.show()

if __name__ == '__main__':
    main()