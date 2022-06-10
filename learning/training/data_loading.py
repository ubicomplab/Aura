import tensorflow as tf

from utils import get_processed_file, NUM_RX_COILS, NUM_TX_COILS
import collections

# Order is important for the csv-readers, so we use an OrderedDict here.
defaults = collections.OrderedDict(
    [("index", [])] +
    [(f"{axis}", []) for axis in "xyz"] +
    [(f"{axis}{i+1}", []) for i in range(NUM_RX_COILS) for axis in "xyz"] +
    [(f"q{axis}", []) for axis in "xyzw"] +
    [(f"t{tx+1}r{rx+1}{axis}", []) for tx in range(NUM_TX_COILS) for rx in range(NUM_RX_COILS) for axis in "xyz"]
    # [(f"pca{i}", []) for i in range(NUM_RX_COILS*9)]
)

pos_labels = ['x', 'y', 'z']
# pos_labels = ['x1', 'y1', 'z1']
rot_labels = ['qw', 'qx', 'qy', 'qz']

SLICE_SIZE = 10000

def dataset(train_file, test_file, variant, train_fraction=0.7, use_random=False):
    def in_training_set(line):
        """Returns a boolean tensor, true if the line is in the training set."""
        # If you randomly split the dataset you won't get the same split in both
        # sessions if you stop and restart training later. Also a simple
        # random split won't work with a dataset that's too big to `.cache()` as
        # we are doing here.
        num_buckets = 1000000
        if use_random:
            bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
            # Use the hash bucket id as a random number that's deterministic per example
            return bucket_id < int(train_fraction * num_buckets)
        # return True
        # index = int(line[:line.index(',')])
        items = tf.decode_csv(line, list(defaults.values()))
        index = items[0]
        return (index % SLICE_SIZE) < (SLICE_SIZE * train_fraction)

    def in_test_set(line):
        """Returns a boolean tensor, true if the line is in the training set."""
        # Items not in the training set are in the test set.
        # This line must use `~` instead of `not` beacuse `not` only works on python
        # booleans but we are dealing with symbolic tensors.
        return ~in_training_set(line)
        # return True

    def decode_line(line):
        """Convert a csv line into a (features_dict,label) pair."""
        # Decode the line to a tuple of items based on the types of
        # csv_header.values().
        items = tf.decode_csv(line, list(defaults.values()))

        # Convert the keys and items to a dict.
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)
        labels_dict = dict()

        # Remove the label from the features_dict
        features_dict.pop('index')
        pos = []
        for y_name in pos_labels:
            # pos.append(features_dict.pop(y_name))
            pos.append(features_dict[y_name])
        labels_dict['pos'] = tf.convert_to_tensor(pos)
        rot = []
        for y_name in rot_labels:
            # rot.append(features_dict.pop(y_name))
            rot.append(features_dict[y_name])
        rot_t = tf.convert_to_tensor(rot)
        labels_dict['rot'] = rot_t

        # m1 = []
        # for i in range(9):
        #     # m.append(features_dict.pop("m%d" % i))
        #     m1.append(features_dict["m1_%d" % i])
        # m1_t = tf.convert_to_tensor(m1)
        # labels_dict['m1'] = m1_t

        # m2 = []
        # for i in range(9):
        #     # m.append(features_dict.pop("m%d" % i))
        #     m2.append(features_dict["m2%d" % i])
        # m2_t = tf.convert_to_tensor(m2)
        # labels_dict['m2'] = m2_t

        # m3 = []
        # for i in range(9):
        #     # m.append(features_dict.pop("m%d" % i))
        #     m3.append(features_dict["m3%d" % i])
        # m3_t = tf.convert_to_tensor(m1)
        # labels_dict['m3'] = m3_t


        # m = []
        # for i in range(9):
        #     # m.append(features_dict.pop("m%d" % i))
        #     m.append(features_dict["m%d" % i])
        # m_t = tf.convert_to_tensor(m)
        # labels_dict['m'] = m_t

        # md = []
        # for i in range(9):
        #     # m.append(features_dict.pop("m%d" % i))
        #     md.append(features_dict["md%d" % i])
        # md_t = tf.convert_to_tensor(md)
        # labels_dict['md'] = md_t

        print(labels_dict)
        return features_dict, labels_dict

    if test_file is None:
        dataset = tf.data.TextLineDataset(get_processed_file(train_file, variant)).skip(1)
        train = (dataset.filter(in_training_set).cache().map(decode_line))
        test = (dataset.filter(in_test_set).cache().map(decode_line))
        full = (dataset.cache().map(decode_line))
    else:
        train_dataset = tf.data.TextLineDataset(get_processed_file(train_file, variant)).skip(1)
        test_dataset = tf.data.TextLineDataset(get_processed_file(test_file, variant)).skip(1)
        train = (train_dataset.cache().map(decode_line))
        test = (test_dataset.cache().map(decode_line))
        full = (test_dataset.cache().map(decode_line))

    return train, test, full
