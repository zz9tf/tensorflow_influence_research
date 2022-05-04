import os.path

from model.matrix_factorization import MF
from model.neural_collaborative_filtering import NCF


class Model():
    """
    Multi-class classification
    """

    def __init__(self, **kwargs):
        self.model = kwargs.pop("model", "MF")
        # create model
        if self.model == "MF":
            self.model = MF(model_configs=kwargs.pop('model_configs'))
        elif self.model == "NCF":
            self.model = NCF(model_configs=kwargs.pop('model_configs'))
        else:
            assert NotImplementedError

        # loading data
        self.dataset = kwargs.pop('dataset')

        # training hyperparameter
        self.batch_size = kwargs.pop('batch_size')
        self.use_batch = kwargs.pop("use_batch", True)
        self.damping = kwargs.pop("damping", 0.0)
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')
        self.decay_epochs = kwargs.pop('decay_epochs')
        self.avextol = kwargs.pop('avextol')
        self.keep_probs = kwargs.pop("keep_probs", None)

        # output/log result position
        self.result_dir = kwargs.pop('result_dir', 'result')
        # make output dictionary
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        #
        self.model_name = kwargs.pop('model_name')
        self.checkpoint_file = os.path.join(self.result_dir,
                                            "%s-checkpoint" % self.model_name)
        print(self.__str__())

    def __str__(self):
        return "Model name: %s\n" % self.model_name\
            + str(self.model) \
            + "-------------------------------\n" \
            + "number of training examples: %d\n" % self.dataset["train"]._x.shape[0] \
            + "number of testing examples: %d\n" % self.dataset["test"]._x.shape[0] \
            + "Using avextol of %.0e\n" % self.avextol \
            + "Using damping of %.0e\n" % self.damping \


    def train(self):
        pass