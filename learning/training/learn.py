import glob

import tensorflow as tf
import numpy as np
from training.data_loading import dataset
from training.estimator import DeepTracker
from utils import save_predictions, load_data, load_norm_data, MAG_RAW_NAMES, NUM_RX_COILS, ALL_MAG_RAW_NAMES, \
    NUM_RX_USED, NUM_TX_USED
import os
import datetime


STEPS = 500000
BASE_LOG_DIR = r'D:\mag_track\models'
TRAIN_DATASET = 'actualCoil'
TEST_DATASET = None
VARIANT = f'3rx'
BATCH_SIZE = 64


PREDICT_ONLY = False
if not PREDICT_ONLY:
    VARIANT += "_shuffle"

def main(argv):
    """Builds, trains, and evaluates the model."""
    assert len(argv) == 1

    # try:
    norm_data = load_norm_data(TRAIN_DATASET, VARIANT)
    # except:
    #     assert False
    #     print("using unit scale")
    #     norm_data = {'pos_scale': np.array([.3,.1,.1])}
    # scale=None
    (train, test, full) = dataset(train_file=TRAIN_DATASET, test_file=TEST_DATASET, variant=VARIANT, train_fraction=0.75, use_random=True)

    def input_train():
        return train.shuffle(50000).batch(BATCH_SIZE).make_one_shot_iterator().get_next()

    def input_test():
        return test.batch(BATCH_SIZE).make_one_shot_iterator().get_next()

    def input_full():
        return full.batch(BATCH_SIZE).make_one_shot_iterator().get_next()

    feature_columns = {}
    for rx_coil in range(NUM_RX_USED):
        for tx_coil in range(NUM_TX_USED):
            for axis in range(3):
                key = ALL_MAG_RAW_NAMES[rx_coil][tx_coil*3 + axis]
                feature_columns[key] = tf.feature_column.numeric_column(key=key)

    # feature_columns = [tf.feature_column.numeric_column(key="pca%d" % i) for i in range(9)]# + \
                      # [tf.feature_column.numeric_column(key="q%s" % i) for i in ['x','y','z','w']]
    # feature_columns = [tf.feature_column.numeric_column(key=i) for i in "xyz"]
    # feature_columns = [tf.feature_column.numeric_column(key="mb%d" % i) for i in range(3)]
    # feature_columns = [tf.feature_column.numeric_column(key=i.replace('_', '')) for i in MAG_RAW_NAMES]
    # feature_columns = [tf.feature_column.numeric_column(key=i.replace('_', '')) for i in MAG_RAW_NAMES[0:3]]
    #                    + \
    # feature_columns = [tf.feature_column.numeric_column(key="pca%d" % i) for i in range(9)] + \
    #                   [tf.feature_column.numeric_column(key=i.replace('_', '')) for i in MAG_RAW_NAMES] + \
    #                   [tf.feature_column.numeric_column(key="ma%d" % i) for i in range(3)] + \
    #                   [tf.feature_column.numeric_column(key="mb%d" % i) for i in range(3)] + \
    #                   [tf.feature_column.numeric_column(key="mc")]
    #                   # [tf.feature_column.numeric_column(key="q%s" % i) for i in ['x','y','z','w']]
    # feature_columns = [tf.feature_column.numeric_column(key=i.replace('_', '')) for i in MAG_RAW_NAMES[0:3]] + \
    #                   [tf.feature_column.numeric_column(key="q%s" % i) for i in ['x','y','z','w']]
    # feature_columns = [tf.feature_column.numeric_column(key=i) for i in ['x', 'y', 'z']] + \
    #                   [tf.feature_column.numeric_column(key="q%s" % i) for i in ['x','y','z','w']]
    # VARIANT =
    key = input("enter model key:") + "_" + TRAIN_DATASET + "_" + VARIANT
    if PREDICT_ONLY:
        key += "_shuffle"
    print(key)
    model_dir = os.path.join(BASE_LOG_DIR, key)

    # for v12 add another 32
    # model = DeepTracker(hidden_units=[64,10], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[64,64,64,64], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[64,64,64,64], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[128,128,128,128], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=.00000001, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[256,256,256,256,256], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=.0000001, norm_data=norm_data)
    model = DeepTracker(hidden_units=[512,512,512,512,512,512], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=.000001, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[512,256,256], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=.000001, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[128,128,256,256], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=.000001, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[256,256], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[64,64,48,48,32,32,32,16,16], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[64,64,48,48,32,32,32,24,24,16,16], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01, norm_data=norm_data)
    # model = DeepTracker(hidden_units=[9,9,9,9,9,9,9,9,9], feature_columns=feature_columns, model_dir=model_dir, initial_learning_rate=0.01)

    # # Train the model.
    # model.train(input_fn=input_train, steps=STEPS)
    #
    # # Evaluate how the model performs on data it has not yet seen.
    # eval_result = model.evaluate(input_fn=input_test)

    train_spec = tf.estimator.TrainSpec(input_fn=input_train, max_steps=STEPS)
    # exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_test, steps=None, exporters=None)

    if not PREDICT_ONLY:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        predictions = model.predict(input_full, predict_keys=['pos', 'rot'])
        all_preds = []
        for p in predictions:
            all_preds.append(np.hstack((p['pos'], p['rot'])))
        all_preds = np.array(all_preds)

        save_predictions(key, TEST_DATASET, np.array(all_preds))


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
