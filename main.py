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
    "batch_size": 2048,  # 3020,  # the batch_size for training or predict, None for not to use batch
    "lr": 1e-3,  # initial learning rate for training MF or NCF model
    "weight_decay": 1e-3,  # l2 regularization term for training MF or NCF model
    # influence function
    "damping": 1e-6,  # damping term in influence function
    "avextol": 1e-3,  # threshold for optimization in influence function
    # train
    "num_epoch_train": 180000,  # training steps
    "load_checkpoint": True,  # whether loading previous model if it exists.
    # retrain
    "num_test": 5,  # number of test points of retraining"
    "retrain_times": 4,
    "num_steps_retrain": 27000,  # retraining steps
    "sort_test_case": 0
}

if configs['dataset'] == 'movielens':
    dataset = load_movielens('./data')
elif configs['dataset'] == 'yelp':
    dataset = load_yelp('./data')
else:
    raise NotImplementedError

num_users = int(np.max(dataset["train"]._x[:, 0]) + 1)
num_items = int(np.max(dataset["train"]._x[:, 1]) + 1)

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

model.train(num_epoch=configs['num_epoch_train'], load_checkpoint=configs['load_checkpoint'])

test_indices = np.random.choice(dataset["test"].num_examples, configs['num_test'], replace=False)
