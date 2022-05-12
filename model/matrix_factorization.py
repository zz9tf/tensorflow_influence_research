import tensorflow as tf
import numpy as np

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
               + "embedding size: %d\n" % self.embedding_size

    def initialize_model(self, seed=0):
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

    def __call__(self, ids):
        """
        This method will do the predict process of the model
        :param ids: A tensor, int 32, represents the selected ids of users and items
        :return: A tensor, float 64, represents the predict result of interaction between special user and item
        """

        user_embedding = tf.nn.embedding_lookup(self.embedding_users, ids[0])
        item_embedding = tf.nn.embedding_lookup(self.embedding_items, ids[1])
        user_bias = tf.nn.embedding_lookup(self.bias_users, ids[0])
        item_bias = tf.nn.embedding_lookup(self.bias_items, ids[1])
        rating = tf.reduce_sum(user_embedding * item_embedding, axis=1) \
                 + user_bias + item_bias + self.global_bias

        return rating

    def get_loss_setting(self):
        """
        This method returns the setting of getting loss method
        :return: A function for getting loss
        """
        return tf.keras.losses.MeanSquaredError()

    def get_optimizer_setting(self, learning_rate=1e-3):
        """
        This method returns the setting of getting optimizer
        :return: An object represents an optimizer
        """
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
