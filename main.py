import numpy as np

from load_data import load_movielens, load_yelp
from model.generic_neural_net import Model

configs = {
    # detaset
    "dataset": "movielens",  # name of dataset: movielens or yelp
    # model configs
    "predict_model": "MF",  # modeltype:MF or NCF
    "embedding_size": 16,  # embedding size
    # train configs
    "batch_size": 4096,  # 3020,  # the batch_size for training or predict, None for not to use batch
    "lr": 1e-3,  # initial learning rate for training MF or NCF model
    "weight_decay": 1e-3,  # l2 regularization term for training MF or NCF model
    # influence function
    "damping": 1e-6,  # damping term in influence function
    "avextol": 1e-3,  # threshold for optimization in influence function
    # train
    "num_epoch_train": 270000,  # training steps
    "load_checkpoint": True,  # whether loading previous model if it exists.
    # Influence on loss by remove one data point
    "num_to_remove": 5,  # the number of data points to be removed to evaluate the influence
    "percentage_to_keep": [0.3, 0.5, 0.7, 0.9],  # A list of the percentage of training dataset to keep
    # retrain
    "num_test": 5,  # number of test points of retraining",
    "retrain_times": 4,
    "sort_test_case": 0
}

if configs['dataset'] == 'movielens':
    dataset = load_movielens('./data')
elif configs['dataset'] == 'yelp':
    dataset = load_yelp('./data')
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

model.train(num_epoch=configs['num_epoch_train']
            , load_checkpoint=configs['load_checkpoint']
            , checkpoint_name="training")

removed_idxs = np.random.choice(dataset["train"].num_examples, configs["num_to_remove"], replace=False)
actual_y_diff = np.zeros(configs["num_to_remove"])
predict_y_diff = np.zeros(configs["num_to_remove"])

for i, removed_idx in enumerate(removed_idxs):
    print("\n======== removed point {}: {} ========".format(i, removed_idx))
    for percentage_to_keep in configs["percentage_to_keep"]:
        print("the percentage of training dataset to keep: %.2f" % percentage_to_keep)
        # training the model with the removed idx
        model.train(
            percentage_to_keep=percentage_to_keep,
            removed_idx=[removed_idx],
            num_epoch=configs["num_epoch_train"],
            load_checkpoint=configs["load_checkpoint"]
        )
        
        # Influence on loss function
        model.predict_x_inf_on_loss_function(removed_idx=removed_idx)


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
        model.train(
            percentage_to_keep=percentage_to_keep,
            num_epoch=configs["num_epoch_train"],
            load_checkpoint=configs["load_checkpoint"]
        )
