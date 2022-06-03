import numpy as np
from scipy.stats import pearsonr

from load_data import load_movielens, load_yelp
from model.matrix_factorization import MF
from model.neural_collaborative_filtering import NCF

configs = {
    # detaset
    "dataset": "movielens",  # name of dataset: movielens or yelp
    # model configs
    "model": "MF",  # modeltype:MF or NCF
    "embedding_size": 16,  # embedding size
    # train configs
    "batch_size": 512,  # 3020,  # the batch_size for training or predict, None for not to use batch
    "lr": 1e-3,  # initial learning rate for training MF or NCF model
    "weight_decay": 1e-3,  # l2 regularization term for training MF or NCF model
    # influence function
    "damping": 1e-6,  # damping term in influence function
    "avextol": 1e-3,  # threshold for optimization in influence function
    # train
    "num_epoch_train": 27000,  # training steps
    "load_checkpoint": False,  # whether loading previous model if it exists.
    "plot": False ,  # if plot the figure of train loss and test loss
    # Influence on single point by remove one data point
    "single_point": ["test", "test_loss"],   # the target y to be evaluated, train_y, train_loss, test_y, test_loss, None. None means not to evaluate.
    "num_of_single": 1,  # the number of data points to be removed to evaluate the influence
    # Influence on loss by remove one data point
    "batch_points": None,  # the target loss function to be evaluated, train, test, None. None means not to evaluate.
    "num_to_removed": 50,  # number of points to retrain
    

    "percentage_to_keep": [1],  # A list of the percentage of training dataset to keep, ex: 0.3, 0.5, 0.7, 0.9
}


if configs['dataset'] == 'movielens':
    dataset = load_movielens('./data')
elif configs['dataset'] == 'yelp':
    dataset = load_yelp('./data')
else:
    raise NotImplementedError

num_users = int(np.max(dataset["train"].x[:, 0]) + 1)
num_items = int(np.max(dataset["train"].x[:, 1]) + 1)

model = None
if configs["model"] == "MF":
    Model = MF
elif model == "NCF":
    Model = NCF
else:
    assert NotImplementedError

model = Model(
    # model
    model_configs={
        'num_users': num_users,
        'num_items': num_items,
        'embedding_size': configs['embedding_size'],
        'weight_decay': configs['weight_decay'],
    },
    basic_configs={
        # loading data
        'dataset': dataset,
        # train configs
        'batch_size': configs['batch_size'],
        'learning_rate': configs['lr'],
        'weight_decay': configs['weight_decay'],
        
        # influence function
        'avextol': configs['avextol'],
        'damping': configs['damping'],
        # loading configs
        'result_dir': 'result',
        'model_name': '%s_%s_explicit_damping%.0e_avextol%.0e_embed%d_wd%.0e' % (
            configs['dataset'], configs['model'], configs['damping']
            , configs['avextol'], configs['embedding_size'], configs['weight_decay'])
    }
)

if configs["single_point"]:
    target_idxs = np.random.choice(dataset[configs["single_point"][0]].num_examples, configs["num_of_single"], replace=False)
    print("target_idx: ", target_idxs)

    actual_singles_diffs = []
    predict_singles_diffs = []

    for percentage_to_keep in configs["percentage_to_keep"]:
        print("the percentage of training dataset to keep: %.2f" % percentage_to_keep)        
        for target_idx in target_idxs:
            # initialize related indexs
            print("target_idx: ", target_idx)
            target_x_idx = model.dataset[configs["single_point"][0]].get_one(target_idx)[0][0]
            related_idxs = model.dataset["train"].get_related_idxs(target_x_idx)
            print("related_idxs: {}\n".format(len(related_idxs)), related_idxs)
            
            # diff space
            actual_single_diff = np.zeros(len(related_idxs))
            predict_single_diff = np.zeros(len(related_idxs))
            
            np.random.seed(17)
            num_to_keep = int(model.dataset["train"].x.shape[0] * percentage_to_keep)
            basic_idxs = np.random.choice(model.dataset["train"].x.shape[0], num_to_keep, replace=False)
            keep_idxs = np.unique(np.concatenate((basic_idxs, np.array([int(idx) for idx in related_idxs[:, 0]]))))
            model.reset_dataset(keep_idxs=keep_idxs)
            # training the original model
            model.train(num_epoch=configs["num_epoch_train"],
                        load_checkpoint=configs["load_checkpoint"],
                        checkpoint_name="orin_{}_{}".format(target_idx, percentage_to_keep),
                        verbose=False,
                        plot=configs["plot"]
                    )
            feed_dict = model.fill_feed_dict(model.dataset[configs["single_point"][0]].get_batch())
            if configs["single_point"][1] in ["test_y", "train_y"]:
                print("get ori y...")
                ori = model.sess.run(model.predicts_op, feed_dict=feed_dict)
            elif configs["single_point"][1] in ["test_loss", "train_loss"]:
                print("get ori loss...")
                ori = model.sess.run(model.loss_op, feed_dict=feed_dict)

            for i, removed_idx in enumerate(related_idxs[:10]):
                print("\n======== removed point {}: {} ========".format(i, removed_idx))
                # reload model
                model.load_model_checkpoint(load_checkpoint=True
                                        , checkpoint_name="orin_{}_{}___{}".format(
                                            target_idx, percentage_to_keep,
                                            model.model_name + "_step{}".format(configs["num_epoch_train"])
                                            ))
                print("loaded model: ", "orin_{}_{}___{}".format(
                                            target_idx, percentage_to_keep,
                                            model.model_name + "_step{}".format(configs["num_epoch_train"])
                                            ))
               
                # Influence on loss function
                predict_single_diff[i] = model.predict_x_inf_on_predict_function(
                                        target_loss=configs["single_point"],
                                        target_id=target_idx,
                                        removed_id=removed_idx
                                        )
                
                # retraining the model without the removed idx
                model.reset_dataset(keep_idxs=keep_idxs, removed_idx=int(removed_idx[0]))
                model.train(num_epoch=configs["num_epoch_train"],
                            load_checkpoint=configs["load_checkpoint"],
                            checkpoint_name="re_{}_{}_{}".format(target_idx, percentage_to_keep, removed_idx),
                            verbose=False,
                            plot=configs["plot"]
                        )
                
                feed_dict = model.fill_feed_dict(model.dataset[configs["single_point"][0]].get_batch())
                if configs["single_point"][1] in ["test_y", "train_y"]:
                    print("get re y...")
                    re = model.sess.run(model.predicts_op, feed_dict=feed_dict)
                elif configs["single_point"][1] in ["test_loss", "train_loss"]:
                    print("get re loss...")
                    re = model.sess.run(model.loss_op, feed_dict=feed_dict)

                print(configs["single_point"][1], "ori: ", ori)
                print(configs["single_point"][1], " re: ", re)
                print("real diff: ", re - ori)
                print("predict diff: ", predict_single_diff[i])
                actual_single_diff[i] = re - ori
            
                actual_singles_diffs.append(actual_single_diff)
                predict_singles_diffs.append(predict_single_diff)

    actual_singles_diffs = np.concatenate(actual_singles_diffs)
    predict_singles_diffs = np.concatenate(predict_singles_diffs)
    print('Correlation is %s' % pearsonr(actual_singles_diffs, predict_singles_diffs)[0])
    np.savez(
        'output/result-%s-%s-%s.npz' % (configs['model'], configs['dataset'] ,configs["single_point"][1]),
        actual_diffs=actual_singles_diffs,
        predict_diffs=predict_singles_diffs,
        target_idxs=target_idxs
    )


if configs["batch_points"] is not None:
    actual_loss_diff = np.zeros(configs["num_to_remove"])
    predict_loss_diff = np.zeros(configs["num_to_remove"])

    for percentage_to_keep in configs["percentage_to_keep"]:
        print("the percentage of training dataset to keep: %.2f" % percentage_to_keep)
        np.random.seed(17)
        num_to_keep = int(model.dataset["train"].x.shape[0] * percentage_to_keep)
        keep_idxs = np.random.choice(model.dataset["train"].x.shape[0], num_to_keep, replace=False)
        # training the original model
        model.reset_dataset(keep_idxs=keep_idxs)
        model.train(
                    num_epoch=configs["num_epoch_train"],
                    load_checkpoint=configs["load_checkpoint"],
                    checkpoint_name="orin_{}".format(percentage_to_keep),
                    verbose=False,
                    plot=configs["plot"]
                )
        feed_dict = model.fill_feed_dict(model.dataset[configs["batch_points"]].get_batch())
        ori_loss = model.sess.run(model.loss_op, feed_dict=feed_dict)

        removed_idxs = np.random.choice(basic_idxs, configs["num_to_remove"], replace=False)
        for i, removed_idx in enumerate(removed_idxs):
            print("\n======== removed point {}: {} ========".format(i, removed_idx))
            # reload model
            model.load_model_checkpoint(load_checkpoint=True
                                    , checkpoint_name="orin_{}___{}".format(
                                        percentage_to_keep,
                                        model.model_name + "_step{}".format(configs["num_epoch_train"])
                                        ))
            print("loaded model: ", "orin_{}___{}".format(
                                        percentage_to_keep,
                                        model.model_name + "_step{}".format(configs["num_epoch_train"])
                                        ))
            # Influence on loss function
            predict_loss_diff[i] = model.predict_x_inf_on_predict_function(removed_id=removed_idx, target_loss=configs["target_loss"])


            # retraining the model without the removed idx
            model.reset_dataset(keep_idxs=keep_idxs, removed_idx=removed_idx)
            model.train(num_epoch=configs["num_epoch_train"],
                            load_checkpoint=configs["load_checkpoint"],
                            checkpoint_name="re_{}_{}".format(percentage_to_keep, removed_idx),
                            verbose=False,
                            plot=configs["plot"]
                        )
            feed_dict = model.fill_feed_dict(model.dataset[configs["batch_points"]].get_batch())
            re_loss = model.sess.run(model.loss_op, feed_dict=feed_dict)
            print("ori_loss: ", ori_loss)
            print("re_loss: ", re_loss)
            print("real diff: ", re_loss - ori_loss)
            print("predict diff: ", predict_loss_diff[i])
            actual_loss_diff[i] = re_loss - ori_loss

    print('Correlation is %s' % pearsonr(actual_loss_diff, predict_loss_diff)[0])
    np.savez(
        'output/result-%s-%s-%s.npz' % (configs['model'], configs['dataset'], "loss_function"),
        actual_diffs=actual_loss_diff,
        predict_diffs=predict_loss_diff,
        target_idxs=removed_idxs
    )

          

