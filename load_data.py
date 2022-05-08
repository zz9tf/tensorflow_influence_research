import numpy as np
import tensorflow as tf

def load_movielens(dir):
  train = np.loadtxt("%s/ml-1m-ex.train.rating"%dir, delimiter='\t')
  valid = np.loadtxt("%s/ml-1m-ex.valid.rating"%dir, delimiter='\t')
  test = np.loadtxt("%s/ml-1m-ex.test.rating"%dir, delimiter='\t')

  train_input = train[:975460,:2].astype(np.int32)
  train_output = train[:975460,2]
  valid_input = valid[:-6, :2].astype(np.int32)
  valid_output = valid[:-6, 2]
  test_input = test[:-6, :2].astype(np.int32)
  test_output = test[:-6, 2]

  train = Dataset(train_input, train_output)
  validation = (valid_input, valid_output)
  test = Dataset(test_input, test_output)

  return {"train": train, "validation": validation, "test": test}

def load_yelp(dir):
  train = np.loadtxt("%s/yelp-ex.train.rating"%dir, delimiter='\t')
  valid = np.loadtxt("%s/yelp-ex.valid.rating"%dir, delimiter='\t')
  test = np.loadtxt("%s/yelp-ex.test.rating"%dir, delimiter='\t')

  train_input = train[:628881,:2].astype(np.int32)
  train_output = train[:628881,2]
  valid_input = valid[:, :2].astype(np.int32)
  valid_output = valid[:, 2]
  test_input = test[:51153, :2].astype(np.int32)
  test_output = test[:51153, 2]

  train = Dataset(train_input, train_output)
  validation = (valid_input, valid_output)
  test = Dataset(test_input, test_output)

  return {"train": train, "validation": validation, "test": test}


class Dataset(object):

    def __init__(self, x, y):

        assert(x.shape[0] == y.shape[0])

        self._x = x
        self._x_batch = np.copy(x)
        self._y = y
        self._y_batch = np.copy(y)
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._num_examples

    def append_one_set(self, case_x, case_y):
        self._x = np.concatenate([self._x, case_x], axis=0)
        self._y = np.concatenate([self._y, case_y], axis=0)
        self._x_batch = np.copy(self._x)
        self._y_batch = np.copy(self._y)
        self._num_examples = self._x.shape[0]
        print("A set is added to the dataset.")
        return self._x.shape[0]-1

    def reset_batch(self):
        self._index_in_epoch = 0
        self._x_batch = np.copy(self._x)
        self._y_batch = np.copy(self._y)

    def get_batch(self, batch_size=None):
        """
        This method will return a batch of data, and move index to the start of next batch
        :param batch_size: A integer represents the batch size
        :return: A tuple contains two tensor list which are batches of x and batches of y
        """
        if batch_size is None:
            batch_size = self.num_examples

        if self._index_in_epoch >= self.num_examples:
            # Shuffle the data
            shuf_index = np.arange(self._num_examples)
            np.random.shuffle(shuf_index)
            self._x_batch = self._x_batch[shuf_index, :]
            self._y_batch = self._y_batch[shuf_index]

            # Start next epoch
            self._index_in_epoch = 0

        start = self._index_in_epoch
        end = min(self._index_in_epoch+batch_size, self.num_examples)
        x_batch = (tf.constant(self._x_batch[start:end, 0]), tf.constant(self._x_batch[start:end, 1]))
        y_batch = tf.constant(self._y_batch[start:end])
        self._index_in_epoch += batch_size

        return x_batch, y_batch