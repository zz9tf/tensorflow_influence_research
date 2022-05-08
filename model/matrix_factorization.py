import tensorflow as tf
import numpy as np
from model.additional_methods import variable

class MF(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(name="matrix_factorization")
        model_configs = kwargs.pop('model_configs')
        self.num_users = model_configs['num_users']
        self.num_items = model_configs['num_items']
        self.embedding_size = model_configs['embedding_size']

        # initialize model
        self.initialize_model()

    def __str__(self):
        return "-----------   MF   ------------\n" \
               + "number of users: %d\n" % self.num_users \
               + "number of items: %d\n" % self.num_items \
               + "embedding size: %d\n" % self.embedding_size \

    def initialize_model(self, seed=0):
        """
        This method initializes the basis parameters in the model, such as
        all parameter including bias for each user and item.
        :return: None
        """
        tf.random.set_seed(seed)  # set seed for initializer
        with tf.name_scope("embedding_layer"):
            self.embedding_users = variable("embedding_users"
                                            , [self.num_users, self.embedding_size]
                                            , tf.keras.initializers.TruncatedNormal(
                                                stddev=1/np.square(self.embedding_size)))

            self.bias_users = variable("bias_users", [self.num_users], tf.constant_initializer(0.0))
            self.embedding_items = variable("embedding_items"
                                            , [self.num_items, self.embedding_size]
                                            , tf.keras.initializers.TruncatedNormal(
                                            stddev=1/np.square(self.embedding_size)))

            self.bias_items = variable("bias_items", [self.num_items], tf.constant_initializer(0.0))
            self.global_bias = variable("global_bias", [1], tf.constant_initializer(0.0))

    def __call__(self, ids):
        """
        This method will do the predict process of the model
        :param user_ids: A tensor, int 32, represents the selected ids of users
        :param item_ids: A tensor, int 32, represents the selected ids of items
        :return: A tensor, float 64, represents the predict result of interaction between special user and item
        """

        user_embedding = tf.nn.embedding_lookup(self.embedding_users, ids[0])
        item_embedding = tf.nn.embedding_lookup(self.embedding_items, ids[1])
        user_bias = tf.nn.embedding_lookup(self.bias_users, ids[0])
        item_bias = tf.nn.embedding_lookup(self.bias_items, ids[1])
        rating = tf.reduce_sum(user_embedding * item_embedding, axis=1)\
            + user_bias + item_bias + self.global_bias

        return rating
