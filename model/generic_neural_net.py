import os.path
import time

import tensorflow as tf
import numpy as np

from model.matrix_factorization import MF
from model.neural_collaborative_filtering import NCF
from model.additional_methods import square_loss


class Model:
    """
    Multi-class classification
    """

    def __init__(self, **kwargs):
        # loading data
        self.dataset = kwargs.pop('dataset')

        # create model
        self.predict = kwargs.pop("predict_model", "MF")
        if self.predict == "MF":
            self.predict = MF(model_configs=kwargs.pop('model_configs'))
        elif self.predict == "NCF":
            self.predict = NCF(model_configs=kwargs.pop('model_configs'))
        else:
            assert NotImplementedError

        # training hyperparameter
        self.batch_size = kwargs.pop('batch_size', None)
        self.weight_decay = kwargs.pop('weight_decay')

        # training model setting
        self.get_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.pop('learning_rate'))

        # influence function
        self.avextol = kwargs.pop('avextol')
        self.damping = kwargs.pop('damping')

        # output/log result position
        self.result_dir = kwargs.pop('result_dir', 'result')
        # make output dictionary
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # loading model
        self.model_name = kwargs.pop('model_name')
        self.checkpoint_file = os.path.join(self.result_dir,
                                            "%s-checkpoint" % self.model_name)
        print(self.__str__())

    def __str__(self):
        return "Model name: %s\n" % self.model_name\
            + str(self.predict) \
            + "weight decay: %d\n" % self.weight_decay \
            + "number of training examples: %d\n" % self.dataset["train"]._x.shape[0] \
            + "number of testing examples: %d\n" % self.dataset["test"]._x.shape[0] \
            + "Using avextol of %.0e\n" % self.avextol \
            + "Using damping of %.0e\n" % self.damping \
            + "-------------------------------\n"

    def train(self, num_epoch, load_checkpoint=False
              , save_checkpoints=True, verbose=True):
        if verbose:
            print("Training for %s epoch" % num_epoch)
        for epoch in range(1, num_epoch):
            start_time = time.time()
            with tf.GradientTape() as tape:
                x, real_y = self.dataset["train"].get_batch(self.batch_size)
                predict_y = self.predict(x)
                loss = self.get_loss(real_y, predict_y)
                gradients = tape.gradient(loss, self.predict.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.predict.trainable_variables))
            duration = time.time() - start_time
            if verbose and epoch % 1000 == 0:
                print('Epoch %d: loss = %.8f (%.3f sec)' % (epoch/1000, loss, duration))

    def get_accuracy(self):
        pass