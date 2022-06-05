import numpy as np
import matplotlib.pyplot as plt

diff_dic = np.load("output/result-MF-movielens-test_y.npz")
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

