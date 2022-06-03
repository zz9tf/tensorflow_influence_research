from torch import embedding
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import numpy as np
import time, os, shutil
from scipy.optimize import fmin_ncg
import matplotlib.pyplot as plt

from model.hessians import hessian_vector_product

tf.disable_v2_behavior()



def split_concatenate_params(concatenate_x, boilerplates):
    """
    This method split all paramters of one product of one user and one item
    """
    split_params = []
    cur_pos = 0
    for boilerplate in boilerplates:
        split_params.append(concatenate_x[cur_pos : cur_pos+boilerplate.shape[0]])
        cur_pos += boilerplate.shape[0]

    return split_params

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

    def __str__(self, details):
        return "Model name: %s\n" % self.model_name \
               + details \
               + "weight decay: %d\n" % self.weight_decay \
               + "number of training examples: %d\n" % self.dataset["train"].num_examples \
               + "number of testing examples: %d\n" % self.dataset["test"].num_examples \
               + "Using avextol of %.0e\n" % self.avextol \
               + "Using damping of %.0e\n" % self.damping \
               + "-------------------------------\n"

    
    def initialize_op(self):
        # Initialize session
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
        self.predicts_op = self.get_predict_op(self.xs_placeholder)
        self.loss_op = self.get_loss_op(self.real_ys_placeholder, self.predicts_op)
        self.one_step_train_op = self.get_one_step_train_op(self.learning_rate, self.loss_op)
        self.fill_feed_dict = lambda batch_data: {self.xs_placeholder: batch_data[0], self.real_ys_placeholder: batch_data[1]}

        self.accuracy_op = self.get_accuracy_op(self.predicts_op, self.real_ys_placeholder)
        self.all_params = self.get_all_params()
        self.grad_all_params_op = tf.gradients(self.loss_op, self.all_params)

    def load_model_checkpoint(self, load_checkpoint=False, checkpoint_name=None):
        self.sess.run(tf.global_variables_initializer())

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
                load_position = os.path.join(self.result_dir, checkpoint_name, "out") 
                self.saver.restore(self.sess, load_position)

        return num_epoch


    def reset_dataset(self, keep_idxs=None, removed_idx=None):
        if removed_idx is not None:
            keep_idxs = keep_idxs[keep_idxs != removed_idx]
        self.dataset["train"].reset_copy(keep_idxs)

        

    def train(self, num_epoch=180000, load_checkpoint=True, save_checkpoints=True
            , verbose=True, checkpoint_name="", plot=False):

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

        self.evaluate()
        
    def retrain(self):
        print("Warning: this method should be override!")

    def evaluate(self):
        """
        This method evaluates the accuracy of the model on training and test sets.
        :return: None
        """
        # Train
        feed_dict = self.fill_feed_dict(self.dataset["train"].get_batch())
        loss = self.sess.run(self.loss_op, feed_dict=feed_dict)
        acc = self.sess.run(self.accuracy_op, feed_dict=feed_dict)
        grads = self.sess.run(self.grad_all_params_op, feed_dict=feed_dict)
        
        # Test
        feed_dict = self.fill_feed_dict(self.dataset["test"].get_batch())
        test_loss, test_acc = self.sess.run([self.loss_op, self.accuracy_op], feed_dict=feed_dict)

        print("\nEvaluation:")
        print('Train loss (w/o reg) on all data: %s' % loss)
        print('Train acc on all data:  %s' % acc)
        print('Test loss (w/o reg) on all data: %s' % test_loss)
        print('Test acc on all data:  %s' % test_acc)
        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(grads)))

    def predict_x_inf_on_loss_function(self, verbose=True, target_loss="train", removed_id=None):
        """
        This method predict the influence of removed single x on training loss function.
        :return:
        """

        assert removed_id is not None
        
        # removed point
        feed_dict = self.fill_feed_dict(self.dataset["train"].get_one(removed_id))
        removed_grads = self.sess.run(self.grad_all_params_op, feed_dict=feed_dict)
        p_placeholder = [tf.placeholder(tf.float64, shape=tf.convert_to_tensor(a).get_shape()) for a in removed_grads]
        hvp_op = hessian_vector_product(self.loss_op, self.all_params, p_placeholder)

        # hvp
        def hvp_f(cg_x):
            cg_x = split_concatenate_params(cg_x, removed_grads)
            feed_dict = self.fill_feed_dict(self.dataset["train"].get_batch())
            for placeholder, x in zip(self.p_placeholder, cg_x):
                feed_dict[placeholder] = x
            return np.concatenate(self.sess.run(hvp_op, feed_dict=feed_dict))

        inverse_hvp = self.get_inverse_hvp(verbose=verbose,
                                        hvp_f=hvp_f,
                                        b=np.concatenate(removed_grads))

        # target loss
        feed_dict = self.fill_feed_dict(self.dataset[target_loss].get_batch())
        target_grads = np.concatenate(self.sess.run(self.grad_all_params_op, feed_dict=feed_dict))

        predict_diff = np.dot(target_grads, inverse_hvp) / self.dataset["train"].num_examples

        return -predict_diff

    def predict_x_inf_on_predict_function(self, verbose=True, target_loss=["test", "test_y"], target_id=None, removed_id=None):
        """
        This method predict the influence of removed single x on the predict value of single test point.
        :return:
        """
        assert target_id is not None
        assert removed_id is not None

        # target point
        feed_dict = self.fill_feed_dict(self.dataset[target_loss[0]].get_one(target_id))
        target_x_id, _ = self.dataset[target_loss[0]].get_one(target_id)
        get_grads = self.create_get_grads(target_x_id[0, 0], target_x_id[0, 1])
        if target_loss[1] in ["test_y", "train_y"]:
            print("get target y...")
            target_grads = get_grads(self.sess.run(tf.gradients(self.predicts_op, self.all_params), feed_dict=feed_dict))
        elif target_loss[1] in ["test_loss", "train_loss"]:
            print("get target loss...")
            target_grads = get_grads(self.sess.run(self.grad_all_params_op, feed_dict=feed_dict))

        p_placeholder = [tf.placeholder(tf.float64, shape=tf.convert_to_tensor(a).get_shape()) for a in target_grads]
        hvp_op = hessian_vector_product(self.loss_op, self.all_params, p_placeholder, get_grads)

        # hvp
        def hvp_f(cg_x):
            cg_x = split_concatenate_params(cg_x, target_grads)
            feed_dict = self.fill_feed_dict(self.dataset["train"].get_batch())
            for placeholder, x in zip(p_placeholder, cg_x):
                feed_dict[placeholder] = x
            return np.concatenate(self.sess.run(hvp_op, feed_dict=feed_dict))

        inverse_hvp = split_concatenate_params(self.get_inverse_hvp(verbose=verbose,
                                                    hvp_f=hvp_f,
                                                    b=np.concatenate(target_grads))
                                                    , target_grads)

        # removed point
        feed_dict = self.fill_feed_dict(self.dataset["train"].get_one(int(removed_id[0])))
        removed_grads = get_grads(self.sess.run(self.grad_all_params_op, feed_dict=feed_dict))

        predict_diff = -(np.dot(np.concatenate(removed_grads), np.concatenate(inverse_hvp))) / self.dataset["train"].num_examples
        # if removed_id[1] == 'u_id':
        #     eu, _, bu, _, gb = removed_grads
        #     hvp_eu, _, hvp_bu, _, hvp_gb = inverse_hvp
        #     predict_diff = -(np.dot(eu, hvp_eu) + np.dot(bu, hvp_bu) + np.dot(gb, hvp_gb)) / self.dataset["train"].num_examples
        # else:
        #     _, ei, _, bi, gb = removed_grads
        #     _, hvp_ei, _, hvp_bi, hvp_gb = inverse_hvp
        #     predict_diff = -(np.dot(ei, hvp_ei) + np.dot(bi, hvp_bi) + np.dot(gb, hvp_gb)) / self.dataset["train"].num_examples

        return predict_diff

    def create_get_grads(self, u_id, i_id):
        def get_grads(ori_grads):
            grads = []
            grads.append(
                ori_grads[0][u_id * self.embedding_size:(1 + u_id) * self.embedding_size])
            grads.append(
                ori_grads[1][i_id * self.embedding_size:(1 + i_id) * self.embedding_size])
            grads.append(
                ori_grads[2][u_id:(1 + u_id)])
            grads.append(
                ori_grads[3][i_id:(1 + i_id)])
            # grads.append(
            #     ori_grads[4]
            # )
            return grads
        return get_grads

    def get_cg_callback(self, verbose, hvp_f, b):
        fmin_loss_split = lambda x: (0.5 * np.dot(hvp_f(x), x), -np.dot(b, x))
        
        def cg_callback(x):
            if verbose:
                half_xhx, bx = fmin_loss_split(x)
                print("Split function value: %s, %s" % (half_xhx, bx))
                print("Function value: ", str(half_xhx + bx))
                print()
        
        return cg_callback

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
