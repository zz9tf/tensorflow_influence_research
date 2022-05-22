from tabnanny import verbose
import time
import tensorflow as tf
import numpy as np
import os
import shutil
from scipy.optimize import fmin_ncg

from model.matrix_factorization import MF
from model.neural_collaborative_filtering import NCF
from model.hessians import hessian_vector_product, get_target_param_grad


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
               + "number of training examples: %d\n" % self.dataset["train"].x.shape[0] \
               + "number of testing examples: %d\n" % self.dataset["test"].x.shape[0] \
               + "Using avextol of %.0e\n" % self.avextol \
               + "Using damping of %.0e\n" % self.damping \
               + "-------------------------------\n"

    def load_model_checkpoint(self, load_checkpoint=False):
        self.predict.initialize_model()
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

    def train(self, random_seed=17, percentage_to_keep=1.0, removed_idx=None, num_epoch=180000
              , load_checkpoint=True, save_checkpoints=True, verbose=True, checkpoint_name=""):
        # select the samples of dataset
        """ This part normally keeps all data and no special data will be kept """
        np.random.seed(random_seed)
        num_to_keep = int(self.dataset["train"].num_examples * percentage_to_keep)
        samples_idxs = np.random.choice(self.dataset["train"].num_examples, num_to_keep, replace=False)
        self.dataset["train"].reset_copy(idxs=samples_idxs, keep_idxs=removed_idx)

        # start training
        """ This part normally initializes the parameter in the model and trains it again """
        checkpoint_name += "___" + self.model_name + "_step%d" % num_epoch
        print("--- Start {} ---".format(checkpoint_name))
        start = self.load_model_checkpoint(load_checkpoint)
        if verbose:
            print("\nTraining for %s epoch" % num_epoch)
        if start == 1:
            for epoch in range(start, num_epoch):
                start_time = time.time()
                with tf.GradientTape() as tape:
                    x_idxs, real_ys = self.dataset["train"].get_batch(self.batch_size)
                    predict_ys = self.predict(x_idxs)
                    loss = self.get_loss(real_ys, predict_ys)
                    gradients = tape.gradient(loss, self.predict.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.predict.trainable_variables))
                duration = time.time() - start_time
                if verbose and epoch % 1000 == 0:
                    print('Epoch %d: loss = %.8f (%.3f sec)' % (epoch, loss, duration))
            if save_checkpoints:
                checkpoint = tf.train.Checkpoint(model=self.predict)
                checkpoint.save(os.path.join(self.result_dir, "out", "out"))
                checkpoint = os.path.join(self.result_dir, checkpoint_name)
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
            x_idxs, real_ys = self.dataset["train"].get_batch()
            predict_ys = self.predict(x_idxs)
            loss = self.get_loss(real_ys, predict_ys)

            test_x_idxs, test_real_ys = self.dataset["test"].get_batch()
            test_predict_ys = self.predict(test_x_idxs)
            test_loss = self.get_loss(test_real_ys, test_predict_ys)

            gradients = tape.gradient(loss, self.predict.trainable_variables)

        print("\nEvaluation:")
        print('Train loss (w/o reg) on all data: %s' % loss.numpy())
        print('Train acc on all data:  %s' % tf.reduce_mean(1 - tf.abs(real_ys - predict_ys)).numpy())

        print('Test loss (w/o reg) on all data: %s' % test_loss.numpy())
        print('Test acc on all data:  %s' % tf.reduce_mean(1 - tf.abs(test_real_ys - test_predict_ys)).numpy())

        gradients = [tf.convert_to_tensor(grad).numpy().flatten() for grad in gradients]
        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(gradients)))

    def predict_x_inf_on_loss_function(self, verbose=True, target_loss="train", removed_idx=None):
        """
        This method predict the influence of removed single x on training loss function.
        :return:
        """

        assert removed_idx is not None
        
        # removed point
        removed_x_idx, removed_y = self.dataset["train"].get_one(removed_idx)
        removed_x0 = self.predict.get_params(removed_x_idx)
        with tf.GradientTape() as tape:
            predict_y = self.predict(removed_x_idx)
            loss = self.get_loss(removed_y, predict_y)
            removed_grad = get_target_param_grad(tape.gradient(loss, self.predict.trainable_variables), removed_x_idx)

        # total loss
        x_idxs, real_ys = self.dataset["train"].get_batch()

        # hvp function
        function_for_hessian = lambda : self.get_loss(real_ys, self.predict(x_idxs))
        hvp_f = lambda cg_x : np.concatenate(hessian_vector_product(xs=self.predict.trainable_variables
                                                                    , function=function_for_hessian
                                                                    , p=self.split_concatenate_params(cg_x)
                                                                    , id=removed_x_idx))
        inverse_hvp = self.get_inverse_hvp(
            verbose=verbose,
            hvp_f=hvp_f,
            b=np.concatenate(removed_grad),
            cg_x0=removed_x0
        )
        print(inverse_hvp)
        # related_idxs = self.dataset[target_loss].get_related_idxs()
        exit()

    def predict_x_inf_on_y_test(self, test_idx=None, removed_idx=None):
        """
        This method predict the influence of removed single x on the predict value of single test point.
        :return:
        """
        assert test_idx is not None
        assert removed_idx is not None
        test_x_idx, test_real_y = self.dataset["test"].get_point(test_idx)
        x_idxs, real_ys = self.dataset["train"].get_batch()
        function = lambda x_idxs : self.get_loss(real_ys, self.predict(x_idxs))
        hessian_vector = hessian_vector_product(x_idxs, function)

    def get_inverse_hvp(self, verbose, hvp_f, b, cg_x0):
        """
        
        """
        fmin_loss_fn = lambda x: 0.5 * np.dot(hvp_f(x), x) - np.dot(b, x)
        fmin_grad_fn = lambda x: hvp_f(x) - b
        fmin_hvp = lambda x, p: hvp_f(p)
        cg_callback = self.get_cg_callback(verbose, fmin_grad_fn, hvp_f, b)
        
        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(cg_x0),
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp,
            callback=cg_callback,
            avextol=self.avextol,
            maxiter=100
        )

        return fmin_results

    def split_concatenate_params(self, concatenate_x):
        """
        This method split all paramters of one product of one user and one item
        """
        split_params = []
        params = self.predict.get_params((0, 0))
        cur_pos = 0
        for param in params:
            split_params.append(concatenate_x[cur_pos : cur_pos+param.shape[0]])
            cur_pos += param.shape[0]

        return split_params
    
    def get_cg_callback(self, verbose, fmin_grad_fn, hvp_f, b):
        fmin_loss_split = lambda x: (0.5 * np.dot(hvp_f(x), x), -np.dot(b, x))
        
        def cg_callback(x):
            if verbose:
                half_xhx, bx = fmin_loss_split(x)
                print("Split function value: %s, %s" % (half_xhx, bx))
                print("Function value: ", str(half_xhx + bx))
                print("Function grad: %s" % fmin_grad_fn(x))
        
        return cg_callback