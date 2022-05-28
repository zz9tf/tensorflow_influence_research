import tensorflow.compat.v1 as tf
import numpy as np
import time, os, shutil
from scipy.optimize import fmin_ncg
import matplotlib.pyplot as plt

from model.hessians import hessian_vector_product, get_target_param_grad
tf.disable_eager_execution()

class Model(object):
    """
    Multi-class classification
    """

    def __init__(self, **kwargs):
        basic_configs = kwargs.pop('basic_configs')
        # loading data
        self.dataset = basic_configs.pop('dataset')

        # training hyperparameter
        self.batch_size = basic_configs.pop('batch_size', None)
        self.learning_rate = basic_configs.pop('learning_rate')
        self.weight_decay = basic_configs.pop('weight_decay')

        # influence function
        self.avextol = basic_configs.pop('avextol')
        self.damping = basic_configs.pop('damping')

        # create loading and saving location
        self.result_dir = basic_configs.pop('result_dir', 'result')
        if os.path.exists(self.result_dir) is False:
            os.makedirs(self.result_dir)
        self.model_name = basic_configs.pop('model_name')

        # Initialize session
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def initialize_op(self):
        self.saver = tf.train.Saver()

        # Setup input
        self.xs_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, 2),
            name="xs_placeholder"
        )
        self.real_ys_placeholder = tf.placeholder(
            tf.float64,
            shape=(None),
            name="real_ys_placeholder"
        )

        # Setup predict and training
        self.predicts_op = self.get_predict(self.xs_placeholder)
        self.loss_op = self.get_loss(self.real_ys_placeholder, self.predicts_op)
        self.all_params = self.get_all_params()
        self.one_step_train_op = self.get_one_step_train_op(self.learning_rate, self.loss_op)
        self.accuracy_op = self.get_accuracy_op(self.predicts_op, self.real_ys_placeholder)
        
        self.train_grad_op = tf.gradients(self.loss_op, self.all_params)
        self.fill_feed_dict = lambda batch_data: {self.xs_placeholder: batch_data[0], self.real_ys_placeholder: batch_data[1]}
        # Setup hessian
        # self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        # self.hvp_op = hessian_vector_product(self.loss_op, self.all_trainable_params, self.p_placeholder)

        self.sess.run(tf.global_variables_initializer())


    def __str__(self, details):
        return "Model name: %s\n" % self.model_name \
               + details \
               + "weight decay: %d\n" % self.weight_decay \
               + "number of training examples: %d\n" % self.dataset["train"].num_examples \
               + "number of testing examples: %d\n" % self.dataset["test"].num_examples \
               + "Using avextol of %.0e\n" % self.avextol \
               + "Using damping of %.0e\n" % self.damping \
               + "-------------------------------\n"

    def load_model_checkpoint(self, load_checkpoint=False, checkpoint_name=None):
        self.initialize_op()

        num_epoch = 1
        if load_checkpoint:
            if checkpoint_name not in os.listdir(os.path.join(self.result_dir)):
                for model in os.listdir(os.path.join(self.result_dir)):
                    print(model)
                checkpoint_name = input("Which model do you want to load?(q to exit)")
            
            if checkpoint_name != "q":
                for char in checkpoint_name.split("_")[-1]:
                    if char.isdigit():
                        num_epoch = num_epoch * 10 + int(char)
                self.saver.restore(self.sess, checkpoint_name)

        return num_epoch

    def train(self, random_seed=17, percentage_to_keep=None, removed_idx=None, num_epoch=180000
              , load_checkpoint=True, save_checkpoints=True, verbose=True, checkpoint_name="", plot=True):
        # select the samples of dataset when percentage_to_keep is not None
        if percentage_to_keep is not None:
            np.random.seed(random_seed)
            num_to_keep = int(self.dataset["train"].x.shape[0] * percentage_to_keep)
            samples_idxs = np.random.choice(self.dataset["train"].x.shape[0], num_to_keep, replace=False)
            self.dataset["train"].reset_copy(idxs=samples_idxs, keep_idxs=removed_idx)
            def retrain():
                self.dataset["train"].reset_copy(idxs=samples_idxs)
                self.train(num_epoch=num_epoch, load_checkpoint=load_checkpoint,
                        checkpoint_name="re_{}_{}".format(removed_idx, percentage_to_keep),
                        plot=plot)
            self.retrain = retrain

        print(self.__str__())

        # start training
        """ This part normally initializes the parameter in the model and trains it """
        checkpoint_name += "___" + self.model_name + "_step%d" % num_epoch
        print("--- Start {} ---".format(checkpoint_name))
        start = self.load_model_checkpoint(load_checkpoint, checkpoint_name)
        if verbose:
            print("\nTraining for %s epoch" % num_epoch)
        if start == 1:
            all_train_loss = []
            all_test_loss = []
            for epoch in range(start, num_epoch):

                start_time = time.time()
                
                # Train
                feed_dict = self.fill_feed_dict(self.dataset["train"].get_batch(self.batch_size))
                _, train_loss = self.sess.run([self.one_step_train_op, self.loss_op], feed_dict=feed_dict)
                all_train_loss.append(train_loss)
                
                # Test
                feed_dict = self.fill_feed_dict(self.dataset["test"].get_batch())
                test_loss = self.sess.run(self.loss_op, feed_dict=feed_dict)
                all_test_loss.append(test_loss)

                duration = time.time() - start_time
                if verbose and epoch % 1000 == 0:
                    print('Epoch %d: loss = %.8f (%.3f sec)' % (epoch, train_loss, duration))

            if save_checkpoints:
                self.saver.save(self.sess, os.path.join(self.result_dir, "out", "out"))
                checkpoint = os.path.join(self.result_dir, checkpoint_name)
                if os.path.exists(checkpoint):
                    shutil.rmtree(checkpoint)
                os.rename(os.path.join(self.result_dir, 'out'), checkpoint)

            if plot:
                plt.plot(all_train_loss, label="train loss")
                plt.plot(all_test_loss, label="test loss")
                plt.legend()
                plt.show()

        # self.evaluate()
        
    def retrain(self):
        print("Warning: this method should be override!")

    def evaluate(self):
        """
        This method evaluates the accuracy of the model on training and test sets.
        :return: None
        """
        with tf.GradientTape() as tape:
            x_idxs, real_ys = self.dataset["train"].get_batch()
            predict_ys = self.model(x_idxs)
            loss = self.get_loss(real_ys, predict_ys)

            test_x_idxs, test_real_ys = self.dataset["test"].get_batch()
            test_predict_ys = self.model(test_x_idxs)
            test_loss = self.get_loss(test_real_ys, test_predict_ys)

            gradients = tape.gradient(loss, self.model.trainable_variables)

        print("\nEvaluation:")
        print('Train loss (w/o reg) on all data: %s' % loss.numpy())
        print('Train acc on all data:  %s' % tf.reduce_mean(1 - tf.abs(real_ys - predict_ys)).numpy())

        print('Test loss (w/o reg) on all data: %s' % test_loss.numpy())
        print('Test acc on all data:  %s' % tf.reduce_mean(1 - tf.abs(test_real_ys - test_predict_ys)).numpy())

        gradients = [tf.convert_to_tensor(grad).numpy().flatten() for grad in gradients]
        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(gradients)))

    def predict_x_inf_on_loss_function(self, verbose=True, target_loss="train", removed_id=None):
        """
        This method predict the influence of removed single x on training loss function.
        :return:
        """

        assert removed_id is not None
        
        # removed point
        removed_x_idx, removed_y = self.dataset["train"].get_one(removed_id)
        with tf.GradientTape() as tape:
            loss = self.get_loss(removed_y, self.model(removed_x_idx))
            removed_grads = [tf.reshape(tf.convert_to_tensor(grad), [-1]) \
                        for grad in tape.gradient(loss, self.model.trainable_variables)]

        # hvp
        x_idxs, real_ys = self.dataset["train"].get_batch()
        function_for_hessian = lambda : self.get_loss(real_ys, self.model(x_idxs))
        hvp_f = lambda cg_x : np.concatenate(hessian_vector_product(xs=self.model.trainable_variables
                                                                    , function=function_for_hessian
                                                                    , ps=self.split_concatenate_params(cg_x, removed_grads)
                                                                    , id=removed_x_idx))
        print(hvp_f(np.concatenate(removed_grads)))
        print(tf.hessians(function_for_hessian(), self.model.trainable_variables))
        exit()

        inverse_hvp = self.get_inverse_hvp(verbose=verbose,
                                        hvp_f=hvp_f,
                                        b=np.concatenate(removed_grads))

        # target loss
        target_x_idxs, target_real_ys = self.dataset[target_loss].get_batch()
        with tf.GradientTape() as tape:
            loss = self.get_loss(target_real_ys, self.model(target_x_idxs))
            target_grads = [tf.reshape(tf.convert_to_tensor(grad), [-1]) \
                        for grad in tape.gradient(loss, self.model.trainable_variables)]
        
        predict_diff = np.dot(np.concatenate(target_grads), inverse_hvp) / self.dataset["train"].num_examples
        return -predict_diff

    def predict_x_inf_on_predict_function(self, test_id=None, removed_id=None):
        """
        This method predict the influence of removed single x on the predict value of single test point.
        :return:
        """
        assert test_id is not None
        assert removed_id is not None
        test_x_idx, test_real_y = self.dataset["test"].get_point(test_id)
        x_idxs, real_ys = self.dataset["train"].get_batch()
        function = lambda x_idxs : self.get_loss(real_ys, self.model(x_idxs))
        hessian_vector = hessian_vector_product(x_idxs, function)

    def get_inverse_hvp(self, verbose, hvp_f, b):
        """
        
        """
        fmin_loss_fn = lambda x: 0.5 * np.dot(x, hvp_f(x)) - np.dot(x, b)
        fmin_grad_fn = lambda x: hvp_f(x) - b
        fmin_hvp = lambda x, p: hvp_f(p)
        cg_callback = self.get_cg_callback(verbose, hvp_f, b)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=b,
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp,
            callback=cg_callback,
            avextol=self.avextol,
            maxiter=100
        )

        return fmin_results

    def split_concatenate_params(self, concatenate_x, boilerplates):
        """
        This method split all paramters of one product of one user and one item
        """
        split_params = []
        cur_pos = 0
        for boilerplate in boilerplates:
            boilerplate = tf.reshape(boilerplate, [-1])
            split_params.append(concatenate_x[cur_pos : cur_pos+boilerplate.shape[0]])
            cur_pos += boilerplate.shape[0]

        return split_params
    
    def get_cg_callback(self, verbose, hvp_f, b):
        fmin_loss_split = lambda x: (0.5 * np.dot(hvp_f(x), x), -np.dot(b, x))
        
        def cg_callback(x):
            if verbose:
                half_xhx, bx = fmin_loss_split(x)
                print("Split function value: %s, %s" % (half_xhx, bx))
                print("Function value: ", str(half_xhx + bx))
                print()
        
        return cg_callback