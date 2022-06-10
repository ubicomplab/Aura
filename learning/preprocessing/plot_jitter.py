import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def main():

    deviations = np.loadtxt(r'D:\mag_track\processed\deviations__t16_precision.csv', delimiter=',')
    print(np.median(deviations))
    sns.set(context="paper", style="whitegrid", font="Lato")
    fig = plt.figure(figsize=(3.33, 2))
    sns.kdeplot(deviations, cumulative=True, lw=3)
    plt.xlabel("Error (mm)")
    plt.ylabel("CDF")
    plt.xlim((0, 2))
    plt.ylim((0, 1))
    fig.subplots_adjust(bottom=0.25, left=0.18)


    plt.show()


if __name__ == "__main__":
    main()
