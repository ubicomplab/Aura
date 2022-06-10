import tensorflow as tf
import numpy as np

from training.rnn_data_loading import rnn_dataset
from training.rnn_estimator import DeepRNNTracker
from utils import save_predictions, load_data, load_norm_data
import os

STEPS = 200000
BASE_LOG_DIR = r'D:\mag_track\models'
DATASET = 't1_march10'  # 'tcombo_shuffle_filt_pcanocube_march1'
BATCH_SIZE = 8
VARIANT = ''
RNN_WINDOW = 128


PREDICT_ONLY = False

def main(argv):
    """Builds, trains, and evaluates the model."""
    assert len(argv) == 1

    norm_data = load_norm_data(DATASET)
    (train, test, full) = rnn_dataset(file=DATASET, rnn_window=RNN_WINDOW)

    def input_train():
        return train.shuffle(512).batch(BATCH_SIZE).make_one_shot_iterator().get_next()

    def input_test():
        return test.batch(BATCH_SIZE).make_one_shot_iterator().get_next()

    def input_full():
        return full.batch(BATCH_SIZE).make_one_shot_iterator().get_next()

    key = input("enter model key:") + "_" + DATASET + "_" + VARIANT
    print(key)
    model_dir = os.path.join(BASE_LOG_DIR, key)

    model = DeepRNNTracker(hidden_units=[64, 48, 48, 32, 32], lstm_hidden_units=64, model_dir=model_dir, initial_learning_rate=0.01, rnn_window=RNN_WINDOW, norm_data=norm_data)

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

        save_predictions(key, np.array(all_preds))



if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
