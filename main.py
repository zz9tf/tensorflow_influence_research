import numpy as np

from load_data import load_movielens, load_yelp
from model.matrix_factorization import MF
from model.neural_collaborative_filtering import NCF
from exprience import single_point_random, single_point_infmax, batch_points_random

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
    "num_epoch_train": 18000,  # training steps
    "load_checkpoint": False,  # whether loading previous model if it exists.
    "plot": False ,  # if plot the figure of train loss and test loss
    # Influence on single point by remove one data point
    "single_point": ["test", "test_y"],   # the target y to be evaluated, train_y, train_loss, test_y, test_loss, None. None means not to evaluate.
    "num_of_single": 50,  # the number of data points to be removed to evaluate the influence
    # Influence on loss by remove one data point
    "batch_points": None,  # the target loss function to be evaluated, train, test, None. None means not to evaluate.
    
    "num_to_removed": 5,  # number of points to retrain
    "retrain_times": 4,  # times to retrain the model
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

# orin_8843_1___movielens_MF_explicit_damping1e-06_avextol1e-03_embed16_wd1e-03_step27000
# model.train(checkpoint_name="test")
# model.predict_x_inf_on_predict_function(target_loss=configs["single_point"],
#                                         target_id=8843,
#                                         removed_id=722042)

if configs["single_point"]:
    single_point_infmax(model, configs)


if configs["batch_points"] is not None:
    batch_points_random(model, configs)