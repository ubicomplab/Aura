import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def main():

    errors = np.loadtxt(r'D:\mag_track\processed\pos_sim.csv', delimiter=',')
    errors_actual = np.loadtxt(r'D:\mag_track\processed\pos_sim_actual.csv', delimiter=',')

    print(np.median(np.abs(errors)))
    print(np.median(np.abs(errors_actual)))
    sns.set(context="paper", style="whitegrid", font="Lato")
    fig = plt.figure(figsize=(3.33, 2))
    sns.kdeplot(np.abs(errors), cumulative=True, lw=3, label="Dipole Model")
    sns.kdeplot(np.abs(errors_actual), cumulative=True, lw=3, label="Prototype coil simulation")
    # sns.kdeplot(np.abs(errors_baseline), cumulative=True, lw=3, label="Baseline")
    plt.xlabel("Error (mm)")
    plt.ylabel("CDF")
    plt.xlim((0, 8))
    plt.ylim((0, 1))
    plt.legend()
    fig.subplots_adjust(bottom=0.25, left=0.18)


    plt.show()


if __name__ == "__main__":
    main()
