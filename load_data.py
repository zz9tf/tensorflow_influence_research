from asyncio import constants
import numpy as np
import tensorflow as tf


def load_movielens(dir):
    train = np.loadtxt("%s/ml-1m-ex.train.rating" % dir, delimiter='\t')
    valid = np.loadtxt("%s/ml-1m-ex.valid.rating" % dir, delimiter='\t')
    test = np.loadtxt("%s/ml-1m-ex.test.rating" % dir, delimiter='\t')

    train_input = train[:975460, :2].astype(np.int32)
    train_output = train[:975460, 2]
    valid_input = valid[:-6, :2].astype(np.int32)
    valid_output = valid[:-6, 2]
    test_input = test[:-6, :2].astype(np.int32)
    test_output = test[:-6, 2]

    train = Dataset(train_input, train_output)
    validation = (valid_input, valid_output)
    test = Dataset(test_input, test_output)

    return {"train": train, "validation": validation, "test": test}


def load_yelp(dir):
    train = np.loadtxt("%s/yelp-ex.train.rating" % dir, delimiter='\t')
    valid = np.loadtxt("%s/yelp-ex.valid.rating" % dir, delimiter='\t')
    test = np.loadtxt("%s/yelp-ex.test.rating" % dir, delimiter='\t')

    train_input = train[:628881, :2]
    train_output = train[:628881, 2]
    valid_input = valid[:, :2]
    valid_output = valid[:, 2]
    test_input = test[:51153, :2]
    test_output = test[:51153, 2]

    train = Dataset(train_input, train_output)
    validation = (valid_input, valid_output)
    test = Dataset(test_input, test_output)

    return {"train": train, "validation": validation, "test": test}


class Dataset(object):

    def __init__(self, x, y):

        assert (x.shape[0] == y.shape[0])

        self.x = x
        self.x_copy = np.copy(x)
        self.y = y
        self.y_copy = np.copy(y)
        self.num_examples = x.shape[0]
        self.index_in_epoch = 0

    def append_one_set(self, case_x, case_y):
        self.x = np.concatenate([self.x, case_x], axis=0)
        self.y = np.concatenate([self.y, case_y], axis=0)
        self.x_copy = np.copy(self.x)
        self.y_copy = np.copy(self.y)
        self.num_examples = self.x.shape[0]
        print("A set is added to the dataset.")
        return self.x.shape[0] - 1

    def reset_copy(self, idxs=None, keep_idxs=None):
        """
        This method will reset the samples in the dataset to be operated.
        :param idxs: A 1d numpy array include all indexs to be kept
        :param keep_idxs: A list includes all special indexs to be kept.
        :return: None
        """
        if idxs is None:
            idxs = np.array(range(self.num_examples))
        if keep_idxs is not None:
            keep_idxs = np.array(keep_idxs)
            idxs = np.concatenate((idxs, keep_idxs), axis=0)
        self.index_in_epoch = 0
        self.x_copy = np.copy(self.x).take(idxs, axis=0)
        self.y_copy = np.copy(self.y).take(idxs)
        self.num_examples = self.x_copy.shape[0]

    def get_batch(self, batch_size=None):
        """
        This method will return a batch of data, and move index to the start of next batch
        :param batch_size: A integer represents the batch size
        :return: A tuple contains two tensor list which are batches of x and batches of y
        """
        if batch_size is None:
            batch_size = self.num_examples

        if self.index_in_epoch >= self.num_examples:
            # Shuffle the data
            shuf_index = np.arange(self.num_examples)
            np.random.shuffle(shuf_index)
            self.x_copy = self.x_copy[shuf_index, :]
            self.y_copy = self.y_copy[shuf_index]

            # Start next epoch
            self.index_in_epoch = 0

        start = self.index_in_epoch
        end = min(self.index_in_epoch + batch_size, self.num_examples)
        x_batch = (tf.constant(self.x_copy[start:end, 0]), tf.constant(self.x_copy[start:end, 1]))
        y_batch = tf.constant(self.y_copy[start:end])
        self.index_in_epoch += batch_size

        return x_batch, y_batch

    def get_one(self, idx=None):
        """
        This method will return one point of data
        """
        assert idx is not None
        x_point = (tf.constant(self.x[idx, 0]), tf.constant(self.x[idx, 1]))
        y_point = tf.constant(self.y[idx])

        return x_point, y_point
