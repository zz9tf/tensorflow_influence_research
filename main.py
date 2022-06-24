import numpy as np
from scipy.stats import pearsonr

from load_data import load_movielens, load_yelp
from model.generic_neural_net import Model

configs = {
    # detaset
    "dataset": "movielens",  # name of dataset: movielens or yelp
    "datapath": "./subdata",  # the path of datasets
    # model configs
    "predict_model": "MF",  # modeltype:MF or NCF
    "embedding_size": 16,  # embedding size
    # train configs
    "batch_size": 512,  # 3020,  # the batch_size for training or predict, None for not to use batch
    "lr": 1e-3,  # initial learning rate for training MF or NCF model
    "weight_decay": 1e-3,  # l2 regularization term for training MF or NCF model
    # influence function
    "damping": 1e-6,  # damping term in influence function
    "avextol": 1e-3,  # threshold for optimization in influence function
    # train
    "num_epoch_train": 10000,  # training steps
    "load_checkpoint": True,  # whether loading previous model if it exists.
    "plot": False,  # if plot the figure of train loss and test loss
    # Influence on loss by remove one data point
    "num_to_remove": 50,  # the number of data points to be removed to evaluate the influence
    "target_loss": "train",  # the target loss function to be evaluated
    "percentage_to_keep": [0.01],  # A list of the percentage of training dataset to keep, ex: 0.3, 0.5, 0.7, 0.9
    # retrain
    "num_test": 5,  # number of test points of retraining",
    "retrain_times": 4,
    "sort_test_case": 0
}

if configs['dataset'] == 'movielens':
    dataset = load_movielens(configs["datapath"])
elif configs['dataset'] == 'yelp':
    dataset = load_yelp(configs["datapath"])
else:
    raise NotImplementedError

num_users = int(np.max(dataset["train"].x[:, 0]) + 1)
num_items = int(np.max(dataset["train"].x[:, 1]) + 1)

model = Model(
    # loading data
    dataset=dataset,
    # model
    model=configs['predict_model'],
    model_configs={
        'num_users': num_users,
        'num_items': num_items,
        'embedding_size': configs['embedding_size'],
        'weight_decay': configs['weight_decay'],
    },
    # train configs
    batch_size=configs['batch_size'],
    learning_rate=configs['lr'],
    weight_decay=configs['weight_decay'],
    # influence function
    avextol=configs['avextol'],
    damping=configs['damping'],
    # loading configs
    result_dir='result',
    model_name='%s_%s_explicit_damping%.0e_avextol%.0e_embed%d_wd%.0e' % (
        configs['dataset'], configs['predict_model'], configs['damping']
        , configs['avextol'], configs['embedding_size'], configs['weight_decay']))

model.train(
            num_epoch=configs["num_epoch_train"],
            load_checkpoint=configs["load_checkpoint"],
            checkpoint_name="test",
            plot=configs["plot"]
        )
model.predict_x_inf_on_loss_function(removed_id=12938, target_loss=configs["target_loss"])


removed_idxs = np.random.choice(dataset["train"].num_examples-1, configs["num_to_remove"], replace=False)
actual_loss_diff = np.zeros(configs["num_to_remove"])
predict_loss_diff = np.zeros(configs["num_to_remove"])

for i, removed_idx in enumerate(removed_idxs):
    print("\n======== removed point {}: {} ========".format(i, removed_idx))
    for percentage_to_keep in configs["percentage_to_keep"]:
        print("the percentage of training dataset to keep: %.2f" % percentage_to_keep)
        # training the model with the removed idx
        total_x_idxs, total_ys = model.dataset[configs["target_loss"]].get_batch()

        model.train(
            percentage_to_keep=percentage_to_keep,
            removed_idx=[removed_idx],
            num_epoch=configs["num_epoch_train"],
            load_checkpoint=configs["load_checkpoint"],
            checkpoint_name="orin_{}_{}".format(removed_idx, percentage_to_keep),
            plot=configs["plot"]
        )
        total_predict_ys = model.predict(total_x_idxs)
        ori_loss = model.get_loss(total_ys, total_predict_ys).numpy()
        print("ori_loss: ", ori_loss)
        # Influence on loss function
        predict_loss_diff[i] = model.predict_x_inf_on_loss_function(removed_id=removed_idx, target_loss=configs["target_loss"])
        # # Influence on y test
        # train_u, train_i = model.data_sets["train"].x[removed_idx]
        # u_idxs = np.where(model.dataset["test"].x[:, 0] == int(train_u))
        # i_idxs = np.where(model.dataset["test"].x[:, 1] == int(train_i))
        # test_influence_idxs = np.concatenate(u_idxs, i_idxs)
        # for test_idx in test_influence_idxs:
        #     model.predict_x_inf_on_y_test(
        #         removed_idx=removed_idx,
        #         test_idx=test_idx
        #     )

        # retraining the model without the removed idx
        model.retrain()

        re_predict_ys = model.predict(total_x_idxs)
        re_loss = model.get_loss(total_ys, re_predict_ys).numpy()
        print("re_loss: ", re_loss)
        print("real diff: ", re_loss - ori_loss)
        print("predict diff: ", predict_loss_diff[i])
        actual_loss_diff[i] = re_loss - ori_loss

print('Correlation is %s' % pearsonr(actual_loss_diff, predict_loss_diff)[0])
np.savez(
    'output/result-%s-%s.npz' % (configs['predict_model'], configs['dataset']),
    actual_loss_diffs=actual_loss_diff,
    predicted_loss_diffs=predict_loss_diff,
    indices_to_remove=removed_idxs
)


