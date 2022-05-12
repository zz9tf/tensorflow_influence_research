import time
import tensorflow as tf
import numpy as np
import os
import shutil

from model.matrix_factorization import MF
from model.neural_collaborative_filtering import NCF


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
        self.get_loss = self.predict.get_loss_setting()
        self.optimizer = self.predict.get_optimizer_setting(learning_rate=kwargs.pop('learning_rate'))

        # influence function
        self.avextol = kwargs.pop('avextol')
        self.damping = kwargs.pop('damping')

        # create loading and saving location
        self.result_dir = kwargs.pop('result_dir', 'result')
        if os.path.exists(self.result_dir) is False:
            os.makedirs(self.result_dir)
        self.model_name = kwargs.pop('model_name')

        print(self.__str__())

    def __str__(self):
        return "Model name: %s\n" % self.model_name \
               + str(self.predict) \
               + "weight decay: %d\n" % self.weight_decay \
               + "number of training examples: %d\n" % self.dataset["train"]._x.shape[0] \
               + "number of testing examples: %d\n" % self.dataset["test"]._x.shape[0] \
               + "Using avextol of %.0e\n" % self.avextol \
               + "Using damping of %.0e\n" % self.damping \
               + "-------------------------------\n"

    def load_model_checkpoint(self, load_checkpoint=False):
        if load_checkpoint:
            print(os.listdir(os.path.join(self.result_dir)))
            model_name = input("Which model do you want to load?(q to exit)")
            num_epoch = 1
            if model_name != "q":
                for char in model_name.split("_")[-1]:
                    if char.isdigit():
                        num_epoch = num_epoch * 10 + int(char)
                checkpoint = tf.train.Checkpoint(model=self.predict)
                checkpoint.restore(os.path.join(self.result_dir, model_name, "out-1"))
            return num_epoch
        else:
            return 1

    def train(self, num_epoch, load_checkpoint=False
              , save_checkpoints=True, verbose=True):
        start = self.load_model_checkpoint(load_checkpoint)
        if verbose:
            print("Training for %s epoch" % num_epoch)
        for epoch in range(start, num_epoch):
            start_time = time.time()
            with tf.GradientTape() as tape:
                x, real_y = self.dataset["train"].get_batch(self.batch_size)
                predict_y = self.predict(x)
                loss = self.get_loss(real_y, predict_y)
                gradients = tape.gradient(loss, self.predict.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.predict.trainable_variables))
            duration = time.time() - start_time
            if verbose and epoch % 1000 == 0:
                print('Epoch %d: loss = %.8f (%.3f sec)' % (epoch, loss, duration))
        if save_checkpoints and start < num_epoch:
            checkpoint = tf.train.Checkpoint(model=self.predict)
            checkpoint.save(os.path.join(self.result_dir, "out", "out"))
            checkpoint = os.path.join(self.result_dir, self.model_name + "_step%d" % num_epoch)
            if os.path.exists(checkpoint):
                shutil.rmtree(checkpoint)
            os.rename(os.path.join(self.result_dir, 'out')
                      , checkpoint)

        self.evaluate()

    def evaluate(self):
        """
        This method evaluates the accuracy of the model on training and test sets.
        :return: None
        """
        with tf.GradientTape() as tape:
            x, real_y = self.dataset["train"].get_batch()
            predict_y = self.predict(x)
            loss = self.get_loss(real_y, predict_y)

            test_x, test_real_y = self.dataset["test"].get_batch()
            test_predict_y = self.predict(test_x)
            test_loss = self.get_loss(test_real_y, test_predict_y)

            gradients = tape.gradient(loss, self.predict.trainable_variables)

        print('Train loss (w/o reg) on all data: %s' % loss.numpy())
        print('Train acc on all data:  %s' % tf.reduce_mean(1-tf.abs(real_y-predict_y)).numpy())

        print('Test loss (w/o reg) on all data: %s' % test_loss.numpy())
        print('Test acc on all data:  %s' % tf.reduce_mean(1 - tf.abs(test_real_y - test_predict_y)).numpy())

        gradients = [tf.convert_to_tensor(grad).numpy().flatten() for grad in gradients]
        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(gradients)))
