import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def main():

    errors = np.loadtxt(r'D:\mag_track\processed\rot_nogravity__t15.csv', delimiter=',')
    errors_grav = np.loadtxt(r'D:\mag_track\processed\rot_gravity__t15.csv', delimiter=',')
    errors_baseline = np.loadtxt(r'D:\mag_track\processed\rot_baseline__t15.csv', delimiter=',')
    print(np.median(np.abs(errors)))
    print(np.median(np.abs(errors_grav)))
    sns.set(context="paper", style="whitegrid", font="Lato")
    fig = plt.figure(figsize=(3.33, 2))
    sns.kdeplot(np.abs(errors), cumulative=True, lw=3, label="Electromagnetic Only")
    sns.kdeplot(np.abs(errors_grav), cumulative=True, lw=3, label="EM + Gravity vector")
    # sns.kdeplot(np.abs(errors_baseline), cumulative=True, lw=3, label="Baseline")
    plt.xlabel("Error (deg)")
    plt.ylabel("CDF")
    plt.xlim((0, 3))
    plt.ylim((0, 1))
    plt.legend()
    fig.subplots_adjust(bottom=0.25, left=0.18)


    plt.show()


if __name__ == "__main__":
    main()
