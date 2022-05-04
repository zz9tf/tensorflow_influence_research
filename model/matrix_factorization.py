import tensorflow as tf

class MF(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MF, self).__init__()
        model_configs = kwargs.pop('model_configs')
        self.num_users = model_configs['num_users']
        self.num_items = model_configs['num_items']
        self.embedding_size = model_configs['embedding_size']
        self.weight_decay = model_configs['weight_decay']

    def __str__(self):
        return "----   MF   ----\n" \
               + "number of users: %d\n" % self.num_users \
               + "number of items: %d\n" % self.num_items \
               + "embedding size: %d\n" % self.embedding_size \
               + "weight decay: %d\n" % self.weight_decay
