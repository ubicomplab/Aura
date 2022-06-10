from preprocessing.step3_feature_extraction import PCA_NAMES
from utils import load_data, OPTI_NAMES, save_data_rnn
import numpy as np

TRIAL = 't1_march10'
RNN_WINDOW = 128
STEP = 8

def main(trial):
    df = load_data(trial, variant="")
    print(df)

    rows = []
    for i in range(0, len(df)-RNN_WINDOW, STEP):
        frame = df.iloc[i:i+RNN_WINDOW]
        data = frame[PCA_NAMES+OPTI_NAMES].as_matrix().flatten()
        rows.append(data)
    all_data = np.stack(rows, axis=0)
    print(all_data)
    save_data_rnn(all_data, TRIAL)


if __name__ == "__main__":
    main(TRIAL)