import tensorflow as tf

from utils import get_processed_file

pos_labels = ['x', 'y', 'z']
rot_labels = ['qw', 'qx', 'qy', 'qz']


SLICE_SIZE = 100

def rnn_dataset(file, train_fraction=0.7, rnn_window=32):
    def in_training_set(line):
        """Returns a boolean tensor, true if the line is in the training set."""
        # If you randomly split the dataset you won't get the same split in both
        # sessions if you stop and restart training later. Also a simple
        # random split won't work with a dataset that's too big to `.cache()` as
        # we are doing here.
        # num_buckets = 1000000
        # bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
        # # Use the hash bucket id as a random number that's deterministic per example
        # return bucket_id < int(train_fraction * num_buckets)
        items = tf.decode_csv(line, [[] for _ in range((9+7)*rnn_window+1)])
        index = items[0]
        return (index % SLICE_SIZE) < (SLICE_SIZE * train_fraction)

    def in_test_set(line):
        """Returns a boolean tensor, true if the line is in the training set."""
        # Items not in the training set are in the test set.
        # This line must use `~` instead of `not` beacuse `not` only works on python
        # booleans but we are dealing with symbolic tensors.
        return ~in_training_set(line)

    def decode_line(line):
        """Convert a csv line into a (features_dict,label) pair."""
        # Decode the line to a tuple of items based on the types of
        # csv_header.values().
        items = tf.decode_csv(line, [[] for _ in range((9+7)*rnn_window+1)])
        reshaped = tf.reshape(items[1:], [-1, rnn_window, (9+7)])
        x = reshaped[0, :, 0:9]
        pos = reshaped[0, :, 9:12]
        rot = reshaped[0, :, 12:]

        labels_dict = dict()
        labels_dict['pos'] = pos
        labels_dict['rot'] = rot


        return x, labels_dict

    dataset = tf.data.TextLineDataset(get_processed_file("rnn_"+file))
    train = (dataset.filter(in_training_set).cache().map(decode_line))
    test = (dataset.filter(in_test_set).cache().map(decode_line))
    full = (dataset.cache().map(decode_line))

    return train, test, full
