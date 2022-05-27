from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

diff_dic = np.load("output/result-MF-movielens.npz")
plt.scatter(diff_dic["actual_loss_diffs"], diff_dic["predicted_loss_diffs"])
min = min(min(diff_dic["actual_loss_diffs"]), min(diff_dic["predicted_loss_diffs"]))
max = max(max(diff_dic["actual_loss_diffs"]), max(diff_dic["predicted_loss_diffs"]))
# plt.plot([min, max], [min, max], color="red")

plt.show()

