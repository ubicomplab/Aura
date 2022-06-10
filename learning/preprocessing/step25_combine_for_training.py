from make_dataset import MAG_NAMES3
from utils import load_resampled_data, MAG_RAW_NAMES, save_data, load_norm_data, load_data, save_resampled_data
import numpy as np
import scipy
import pandas as pd


TRIALS = ['trainA_march28', 'trainB_march28']
# TRIALS = ['t1_march10', 't3_march10']
VARIANTS = ["", "nohigh", "nohighnolow", 'onlybest']

NEW_TRIAL = 'trainAB_march28'
# NEW_TRIAL = 't13_march10'


def main():
    for variant in VARIANTS:
        dfs = []
        for trial in TRIALS:
            partial_df = load_resampled_data(trial, variant=variant)
            dfs.append(partial_df)
        df = pd.concat(dfs, ignore_index=True)
        save_resampled_data(df, NEW_TRIAL, variant=variant)


if __name__ == "__main__":
    main()
