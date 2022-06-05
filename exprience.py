import numpy as np
from scipy.stats import pearsonr

def single_point_random(model, configs):
    target_idxs = np.random.choice(model.dataset[configs["single_point"][0]].num_examples, configs["num_of_single"], replace=False)
    print("target_idx: ", target_idxs)

    actual_singles_diffs = []
    predict_singles_diffs = []

    for percentage_to_keep in configs["percentage_to_keep"]:
        print("the percentage of training dataset to keep: %.2f" % percentage_to_keep)        
        for target_idx in target_idxs:
            # initialize related indexs
            print("target_idx: ", target_idx)
            target_x_idx, target_real_y = model.dataset[configs["single_point"][0]].get_by_idxs(target_idx)
            related_idxs = model.dataset["train"].get_related_idxs(target_x_idx[0])
            print("related_idxs: {}\n".format(len(related_idxs)), related_idxs)
            
            np.random.seed(17)
            num_to_keep = int(model.dataset["train"].x.shape[0] * percentage_to_keep)
            basic_idxs = np.random.choice(model.dataset["train"].x.shape[0], num_to_keep, replace=False)
            keep_idxs = np.unique(np.concatenate((basic_idxs, related_idxs)))
            model.reset_dataset(keep_idxs=keep_idxs)
            # training the original model
            model.train(num_epoch=configs["num_epoch_train"],
                        load_checkpoint=configs["load_checkpoint"],
                        checkpoint_name="orin_{}_{}".format(target_idx, percentage_to_keep),
                        verbose=True,
                        plot=configs["plot"])
            
            feed_dict = model.fill_feed_dict((target_x_idx, target_real_y))
            if configs["single_point"][1] in ["test_y", "train_y"]:
                print("get ori y...")
                ori = model.sess.run(model.predicts_op, feed_dict=feed_dict)
            elif configs["single_point"][1] in ["test_loss", "train_loss"]:
                print("get ori loss...")
                ori = model.sess.run(model.loss_op, feed_dict=feed_dict)
            else:
                assert NotImplementedError

            for i, removed_idx in enumerate(related_idxs[:20]):
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
                predict_singles_diffs.append(model.predict_x_inf_on_predict_function(
                                            target_loss=configs["single_point"],
                                            target_id=target_idx,
                                            removed_id=removed_idx,
                                            related_idxs=related_idxs
                                            ))
                
                # retraining the model without the removed idx
                model.reset_dataset(keep_idxs=keep_idxs, removed_idx=removed_idx)
                re = []
                for i in range(configs["retrain_times"]):
                    model.train(num_epoch=configs["num_epoch_train"],
                                load_checkpoint=configs["load_checkpoint"],
                                checkpoint_name="{}re_{}_{}_{}".format(i, target_idx, percentage_to_keep, removed_idx),
                                verbose=False,
                                plot=configs["plot"]
                            )
                    
                    if configs["single_point"][1] in ["test_y", "train_y"]:
                        print("get re y...")
                        re.append(model.sess.run(model.predicts_op, feed_dict=feed_dict))
                    elif configs["single_point"][1] in ["test_loss", "train_loss"]:
                        print("get re loss...")
                        re.append(model.sess.run(model.loss_op, feed_dict=feed_dict))
                    else:
                        assert NotImplementedError
                
                actual_singles_diffs.append(sum(re)/len(re) - ori)

                print(configs["single_point"][1], " ori: ", ori)
                print(configs["single_point"][1], " re: ", re)
                print("real diff: ", actual_singles_diffs[-1])
                print("predict diff: ", predict_singles_diffs[-1])
    actual_singles_diffs = np.array(actual_singles_diffs)
    predict_singles_diffs = np.array(predict_singles_diffs)
    print('Correlation is %s' % pearsonr(actual_singles_diffs, predict_singles_diffs)[0])
    np.savez(
        'output/%s-%s-%s-random.npz' % (configs['model'], configs['dataset'] ,configs["single_point"][1]),
        actual_diffs=actual_singles_diffs,
        predict_diffs=predict_singles_diffs,
        target_idxs=target_idxs
    )

def single_point_infmax(model, configs):
    target_idxs = np.random.choice(model.dataset[configs["single_point"][0]].num_examples, configs["num_of_single"], replace=False)
    print("target_idx: ", target_idxs)

    actual_singles_diffs = []
    predict_singles_diffs = []

    for percentage_to_keep in configs["percentage_to_keep"]:
        print("the percentage of training dataset to keep: %.2f" % percentage_to_keep)        
        for target_idx in target_idxs:
            # initialize related indexs
            print("target_idx: ", target_idx)
            target_x_idx, target_real_y = model.dataset[configs["single_point"][0]].get_by_idxs(target_idx)
            related_idxs = model.dataset["train"].get_related_idxs(target_x_idx[0])
            print("related_idxs: {}\n".format(len(related_idxs)), related_idxs)
            
            np.random.seed(17)
            num_to_keep = int(model.dataset["train"].x.shape[0] * percentage_to_keep)
            basic_idxs = np.random.choice(model.dataset["train"].x.shape[0], num_to_keep, replace=False)
            keep_idxs = np.unique(np.concatenate((basic_idxs, related_idxs)))
            model.reset_dataset(keep_idxs=keep_idxs)
            # training the original model
            model.train(num_epoch=configs["num_epoch_train"],
                        load_checkpoint=configs["load_checkpoint"],
                        checkpoint_name="orin_{}_{}".format(target_idx, percentage_to_keep),
                        verbose=True,
                        plot=configs["plot"])
            
            feed_dict = model.fill_feed_dict((target_x_idx, target_real_y))
            if configs["single_point"][1] in ["test_y", "train_y"]:
                print("get ori y...")
                ori = model.sess.run(model.predicts_op, feed_dict=feed_dict)
            elif configs["single_point"][1] in ["test_loss", "train_loss"]:
                print("get ori loss...")
                ori = model.sess.run(model.loss_op, feed_dict=feed_dict)
            else:
                assert NotImplementedError
            

            predict_single_diffs = np.zeros([len(related_idxs)])
            for i, removed_idx in enumerate(related_idxs):
                # Influence on loss function
                predict_single_diffs[i] = model.predict_x_inf_on_predict_function(
                                            target_loss=configs["single_point"],
                                            target_id=target_idx,
                                            removed_id=removed_idx,
                                            related_idxs=related_idxs
                                            )
            
            sorted_predict_idxs =  np.argsort(-np.abs(predict_single_diffs))
            related_idxs = related_idxs[sorted_predict_idxs]
            predict_single_diffs = predict_single_diffs[sorted_predict_idxs]
            for i in range(configs["num_to_removed"]):
                # retraining the model without the removed idx
                model.reset_dataset(keep_idxs=keep_idxs, removed_idx=related_idxs[i])
                re = []
                for i in range(configs["retrain_times"]):
                    model.train(num_epoch=configs["num_epoch_train"],
                                load_checkpoint=configs["load_checkpoint"],
                                checkpoint_name="{}re_{}_{}_{}".format(i, target_idx, percentage_to_keep, removed_idx),
                                verbose=False,
                                plot=configs["plot"]
                            )
                    
                    if configs["single_point"][1] in ["test_y", "train_y"]:
                        print("get re y...")
                        re.append(model.sess.run(model.predicts_op, feed_dict=feed_dict))
                    elif configs["single_point"][1] in ["test_loss", "train_loss"]:
                        print("get re loss...")
                        re.append(model.sess.run(model.loss_op, feed_dict=feed_dict))
                    else:
                        assert NotImplementedError
                
                actual_singles_diffs.append(sum(re)/len(re) - ori)
                predict_singles_diffs.append(predict_single_diffs[i])

                print(configs["single_point"][1], " ori: ", ori)
                print(configs["single_point"][1], " re: ", re)
                print("real diff: ", actual_singles_diffs[-1])
                print("predict diff: ", predict_singles_diffs[-1])
    actual_singles_diffs = np.array(actual_singles_diffs)
    predict_singles_diffs = np.array(predict_singles_diffs)
    print('Correlation is %s' % pearsonr(actual_singles_diffs, predict_singles_diffs)[0])
    np.savez(
        'output/%s-%s-%s-maxinf.npz' % (configs['model'], configs['dataset'] ,configs["single_point"][1]),
        actual_diffs=actual_singles_diffs,
        predict_diffs=predict_singles_diffs,
        target_idxs=target_idxs
    )

def batch_points_random(model, configs):
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

        removed_idxs = np.random.choice(keep_idxs, configs["num_to_remove"], replace=False)
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

          
