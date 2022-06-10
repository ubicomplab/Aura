from make_dataset import extract_coordinates, normalize, do_pca, smooth_data, smooth_pca, PCA_NAMES
import pandas as pd

from utils import save_data, save_data_rnn, OPTI_NAMES
import numpy as np

KEYS = ['t%d_march1' % x for x in range(1,11)]
KEYS_ERIC = ['t%d_march1' % x for x in [1,2,5,7,9]]
KEYS_FARSHID = ['t%d_march1' % x for x in [3,4,6,8,10]]
KEYS_EASY = ['t%d_march1' % x for x in [1,2,3,4,10]]



MARCH5_KEYS = ['t%d_march5' % x for x in [1,3]]

COMBO_KEY = 'lstm16_eric_march5'


RNN_WINDOW = 16

def main():
    frames = []
    for key in MARCH5_KEYS:
        print(key)
        frames.append(extract_coordinates(key, drop_high=False, drop_low=False))

    df = pd.concat(frames)

    # smooth_data(df)
    do_pca(df)
    # smooth_pca(df)
    normalize(df)

    features = df[PCA_NAMES].as_matrix()
    labels = df[OPTI_NAMES].as_matrix()
    all_data = []
    all_labels = []
    for frame in frames:
        split_features = np.take(features, range(len(frame)), axis=0)
        split_labels = np.take(labels, range(len(frame)), axis=0)
        print(split_features.shape)
        adj_len = (split_features.shape[0] // RNN_WINDOW) * RNN_WINDOW
        batched_data = np.array(np.split(split_features[:adj_len,:], adj_len/RNN_WINDOW, axis=0))
        batched_labels = np.array(np.split(split_labels[:adj_len,:], adj_len/RNN_WINDOW, axis=0))
        print(batched_data.shape)
        all_data.append(batched_data)
        all_labels.append(batched_labels)
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)

    all_data = np.concatenate((all_data, all_labels), axis=2)
    print(all_data.shape)

    flattened = all_data.reshape((-1, RNN_WINDOW*(9+7)))

    save_data_rnn(flattened, COMBO_KEY)


if __name__ == "__main__":
    main()