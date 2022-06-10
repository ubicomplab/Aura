from make_dataset import extract_coordinates, normalize, do_pca, smooth_data, smooth_pca
import pandas as pd

from utils import save_data

KEYS = ['t%d_march1' % x for x in range(1,11)]
KEYS_ERIC = ['t%d_march1' % x for x in [1,2,5,7,9]]
KEYS_FARSHID = ['t%d_march1' % x for x in [3,4,6,8,10]]
KEYS_EASY = ['t%d_march1' % x for x in [1,2,3,4,10]]



MARCH5_KEYS = ['t%d_march10' % x for x in [1,3]]

COMBO_KEY = 't13_eric_filt_noclip_march10'



def main():
    frames = []
    for key in MARCH5_KEYS:
        print(key)
        frames.append(extract_coordinates(key, drop_high=True))

    df = pd.concat(frames)

    # smooth_data(df)
    do_pca(df)
    # smooth_pca(df)
    normalize(df)

    save_data(df, COMBO_KEY)
    df_sampled = df.sample(frac=1).reset_index(drop=True)
    save_data(df_sampled, COMBO_KEY+"_shuffle")

    # smooth_data(df)
    # save_data(df, COMBO_KEY+"_filtered")


if __name__ == "__main__":
    main()