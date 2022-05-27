import tensorflow as tf
import numpy as np
import os
from model.generic_neural_net import Model

class MF(Model):
    def __init__(self, **kwargs):
        super().__init__(name="matrix_factorization")
        model_configs = kwargs.pop('model_configs')
        self.num_users = model_configs['num_users']
        self.num_items = model_configs['num_items']
        self.embedding_size = model_configs['embedding_size']
        self.weight_decay = model_configs['weight_decay']


    def __str__(self):
        return "-----------   MF   ------------\n" \
               + "number of users: %d\n" % self.num_users \
               + "number of items: %d\n" % self.num_items \
               + "embedding size: %d\n" % self.embedding_size

    def initialize_parameters(self, seed=0):
        
        """
        This method initializes the basis parameters in the model, such as
        all parameter including bias for each user and item.
        :return: None
        """

        tf.random.set_seed(seed)  # set seed for initializer
        with tf.name_scope("embedding_layer"):
            initializer = tf.keras.initializers.TruncatedNormal(
                stddev=1 / np.square(self.embedding_size))
            self.embedding_users = tf.Variable(initial_value=initializer(
                shape=[self.num_users, self.embedding_size]
                , dtype=tf.float64)
                , name="embedding_users")
            self.embedding_items = tf.Variable(initial_value=initializer(
                shape=[self.num_items, self.embedding_size]
                , dtype=tf.float64)
                , name="embedding_items")

            initializer = tf.constant_initializer(0.0)
            self.bias_users = tf.Variable(initial_value=initializer(
                shape=[self.num_users]
                , dtype=tf.float64)
                , name="bias_users")
            self.bias_items =tf.Variable(initial_value=initializer(
                shape=[self.num_items]
                , dtype=tf.float64)
                , name="bias_items")
            self.global_bias = tf.Variable(
                initial_value=initializer(
                shape=[1]
                , dtype=tf.float64)
                , name="global_bias")

    def get_predict(self, ids):
        
        """
        This method will do the predict process of the model
        :param ids: A tensor, float 64, represents the selected ids of users and items
        :return: A tensor, float 64, represents the predict result of interaction between special user and item
        """
        user_embedding = tf.nn.embedding_lookup(self.embedding_users, ids[0])
        item_embedding = tf.nn.embedding_lookup(self.embedding_items, ids[1])
        user_bias = tf.nn.embedding_lookup(self.bias_users, ids[0])
        item_bias = tf.nn.embedding_lookup(self.bias_items, ids[1])

        if len(ids[0].shape) == 0:  # for single point
            rating = user_embedding * item_embedding + user_bias + item_bias \
                 + self.global_bias
        else:
            rating = tf.reduce_sum(user_embedding * item_embedding, axis=1) \
                 + user_bias + item_bias + self.global_bias

        return rating

    def get_loss(self, real_y, predict_y):
        """
        This method returns the loss """
        loss_val = tf.reduce_mean((real_y - predict_y)**2)
        if self.weight_decay is not None:
            loss_val += tf.math.multiply(tf.nn.l2_loss(self.embedding_users), self.weight_decay)
            loss_val += tf.math.multiply(tf.nn.l2_loss(self.embedding_items), self.weight_decay)
        return loss_val

    def get_one_step_train_op(self, loss_op, learning_rate=1e-3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        one_step_train_op = optimizer.minimize(loss_op)
        
        return one_step_train_op

    def get_accuracy_op(self, predict_op, real_ys):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          predict_op: Logits tensor, float - [batch_size, NUM_CLASSES].
          real_ys: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # correct = tf.nn.in_top_k(logits, labels, 1)
        # return tf.reduce_mean(1. - tf.abs(logits - labels) / (labels + 0.0001))
        return tf.reduce_mean(tf.abs(predict_op - real_ys))

    def get_params(self, ids=None):
        """ waiting to be modified """
        if ids is None:
            params = [self.bias_items, self.bias_users, self.embedding_items, self.embedding_users] #, self.global_bias
            return [tf.reshape(param, [-1]) for param in params]

        user_embedding = tf.nn.embedding_lookup(self.embedding_users, ids[0])
        item_embedding = tf.nn.embedding_lookup(self.embedding_items, ids[1])
        user_bias = tf.nn.embedding_lookup(self.bias_users, ids[0])
        item_bias = tf.nn.embedding_lookup(self.bias_items, ids[1])
        params = [item_bias, user_bias, item_embedding, user_embedding] # , self.global_bias

        return [tf.reshape(param, [-1]) for param in params]