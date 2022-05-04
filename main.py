import numpy as np

from load_data import load_movielens, load_yelp
from model.generic_neural_net import Model

configs = {
    "model": "MF",  # modeltype:MF or NCF
    # model configs
    "num_users": None,  # the amount of all different users
    "num_items": None,  # the amount of all different items
    "embedding_size": 16,  # embedding size
    "weight_decay": 1e-3,  # l2 regularization term for training MF or NCF model
    # detaset
    "dataset": "movielens",  # name of dataset: movielens or yelp
    # train configs
    "batch_size": None,  # the size of each batch determined by dataset
    "damping": 1e-6,  # damping term in influence function
    "lr": 1e-3,  # initial learning rate for training MF or NCF model
    "decay_epochs": [10000, 20000],
    "avextol": 1e-3,  # threshold for optimization in influence function
    "load_checkpoint": 1,
    "num_steps_train": 180000,  # training steps
    "reset_adam": 0,
    "maxinf": 1,
    # retrain
    "num_test": 5,  # number of test points of retraining"
    "retrain_times": 4,
    "num_steps_retrain": 27000,  # retraining steps
    "sort_test_case": 0
}

if configs['dataset'] == 'movielens':
    dataset = load_movielens('./data')
    configs['batch_size'] = 3020
elif configs['dataset'] == 'yelp':
    dataset = load_yelp('./data')
    configs['batch_size'] = 3009
else:
    raise NotImplementedError


configs['num_users'] = int(np.max(dataset["train"]._x[:, 0]) + 1)
configs['num_items'] = int(np.max(dataset["train"]._x[:, 1]) + 1)

model = Model(
    # model
    model=configs['model'],
    model_configs={
        'num_users': configs['num_users'],
        'num_items': configs['num_items'],
        'embedding_size': configs['embedding_size'],
        'weight_decay': configs['weight_decay']
    },
    dataset=dataset,
    # train configs
    batch_size=configs['batch_size'],
    use_batch=True,
    damping=configs['damping'],
    initial_learning_rate=configs['lr'],
    decay_epochs=configs['decay_epochs'],
    avextol=configs['avextol'],
    # loading configs
    result_dir='result',
    model_name='%s_%s_explicit_damping%.0e_avextol%.0e_embed%d_maxinf%d_wd%.0e' % (
        configs['dataset'], configs['model'], configs['damping'], configs['avextol']
        , configs['embedding_size'], configs['maxinf'], configs['weight_decay']))

# model.train(num_steps=configs['num_steps_train'], load_checkpoints=True)

test_indices = np.random.choice(dataset["test"].num_examples, configs['num_test'], replace=False)