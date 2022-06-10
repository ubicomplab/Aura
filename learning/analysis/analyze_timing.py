from utils import load_segmented_data, load_extracted_opti_data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

TRIAL = "trainA_march28"

def main(key):
    data_raw = load_segmented_data(key)  # loads the raw high speed magnetic file
    opti_raw = load_extracted_opti_data(key)  # loads the raw high speed opti file

    dt_mag = np.diff(data_raw.time.values)
    dt_opti = np.diff(opti_raw.time.values)
    plt.figure()
    plt.plot(dt_mag)
    plt.figure()
    plt.plot(dt_opti)
    plt.show()


if __name__ == "__main__":
    main(TRIAL)
