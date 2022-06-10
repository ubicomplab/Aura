from make_dataset import MAG_NAMES3
from utils import load_resampled_data, MAG_RAW_NAMES, save_data, load_norm_data, load_data, save_resampled_data
import numpy as np
import scipy
import pandas as pd


TRIAL = 'test_worn_march28'
# TRIALS = ['t1_march10', 't3_march10']
VARIANTS = ["", "nohigh", "nohighnolow", 'onlybest']



def main():
    for variant in VARIANTS:
        df = load_resampled_data(TRIAL, variant=variant)
        msk = np.random.rand(len(df)) < 0.5
        dfA = df[msk]
        dfB = df[~msk]
        save_resampled_data(dfA, TRIAL+"A", variant=variant)
        save_resampled_data(dfB, TRIAL+"B", variant=variant)


if __name__ == "__main__":
    main()
