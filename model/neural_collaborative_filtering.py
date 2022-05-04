import tensorflow as tf

class NCF(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NCF, self).__init__()
        model_configs = kwargs.pop('model_configs')
        self.num_users = model_configs['num_users']
        self.num_items = model_configs['num_items']
        self.embedding_size = model_configs['embedding_size']
        self.weight_decay = model_configs['weight_decay']

    def __str__(self):
        return "----   NCF   ----\n" + \
               "number of users: %d" % self.num_users + \
               "number of items: %d" % self.num_items