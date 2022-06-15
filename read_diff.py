import numpy as np
import matplotlib.pyplot as plt

diff_dic = np.load("result/output/RQ1-MF-movielens.npz", allow_pickle=True)

for act_dif, pre_dif, remove_id in zip(diff_dic["actual_loss_diffs"], diff_dic["predicted_loss_diffs"], diff_dic["indices_to_remove"]):
    plt.scatter(act_dif, pre_dif)
    all_min = min(min(act_dif), min(pre_dif))
    all_max = max(max(act_dif), max(pre_dif))
    plt.plot([all_min, all_max], [all_min, all_max], color="blue", label="x=y")
    actual_min = np.min(act_dif)
    actual_max = np.max(act_dif)
    predict_min = np.min(pre_dif)
    predict_max = np.max(pre_dif)
    plt.plot([actual_min, actual_max], [predict_min, predict_max], color="red", label="min2max on two D")
    plt.plot([actual_min, actual_max], [0, 0], color="orange", label="predict effect dividing line")
    plt.plot([0, 0], [predict_min, predict_max], color="orange", label="actual effect dividing line")
    plt.xlabel("actuall diff")
    plt.ylabel("predict diff")
    plt.legend()
    plt.show()

exit()

plt.scatter(diff_dic["actual_diffs"], diff_dic["predict_diffs"])
min = min(min(diff_dic["actual_diffs"]), min(diff_dic["predict_diffs"]))
max = max(max(diff_dic["actual_diffs"]), max(diff_dic["predict_diffs"]))
# plt.plot([min, max], [min, max], color="red")
actual_min = np.min(diff_dic["actual_diffs"])
actual_max = np.max(diff_dic["actual_diffs"])
predict_min = np.min(diff_dic["predict_diffs"])
predict_max = np.max(diff_dic["predict_diffs"])
plt.plot([actual_min, actual_max], [predict_min, predict_max], color="red")
plt.plot([actual_min, actual_max], [0, 0], color="orange")
plt.plot([0, 0], [predict_min, predict_max], color="orange")
plt.xlabel("actuall diff")
plt.ylabel("predict diff")
plt.show()

