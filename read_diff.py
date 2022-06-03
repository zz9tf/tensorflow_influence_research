from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

diff_dic = np.load("output/result-MF-movielens-test_y.npz")
plt.scatter(diff_dic["actual_diffs"], diff_dic["predict_diffs"])
min = min(min(diff_dic["actual_diffs"]), min(diff_dic["predict_diffs"]))
max = max(max(diff_dic["actual_diffs"]), max(diff_dic["predict_diffs"]))
plt.plot([min, max], [min, max], color="red")

plt.show()

