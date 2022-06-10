from utils import load_data, save_data
import numpy as np

KEY = 'combo_demean_march1'

NUM_TO_TAKE = 100


BINS = np.hstack(([-100], np.linspace(-.6,.6,10, endpoint=True), [100]))
print(BINS)

def main():
    print("loading")
    df = load_data(KEY+"_shuffle")
    print("plotting")
    df['x'].plot.hist()
    df['y'].plot.hist()
    df['z'].plot.hist()
    print("digitizing")

    df['x_bin'] = np.digitize(df['x'], BINS)
    df['y_bin'] = np.digitize(df['y'], BINS)
    df['z_bin'] = np.digitize(df['z'], BINS)

    print("grouping")
    df_group = df.groupby(['x_bin', 'y_bin', 'z_bin'], group_keys=False)
    print(df_group.count())
    stratified = df_group.apply(lambda x: x.sample(min(len(x), NUM_TO_TAKE)))
    print(stratified.columns)
    stratified.drop(columns=['x_bin', 'y_bin', 'z_bin'], inplace=True)
    stratified.reset_index(inplace=True, drop=True)
    print(stratified)

    save_data(stratified, KEY+"_shuffle_strat")

if __name__ == "__main__":
    main()