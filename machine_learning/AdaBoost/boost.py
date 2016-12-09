import numpy as np


def stump_classify(data_mat, dimension, threshold_val, threshold_ineq):
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    if threshold_ineq == "lt":
        ret_array[data_mat[:, dimension] <= threshold_val] = -1.0
    else:
        ret_array[data_mat[:, dimension] > threshold_val] = 1.0
    return ret_array


def build_stump(data_arr, class_labels, weight):
    data_mat = np.matrix(data_arr)
    m, n = data_mat.shape
    label_mat = np.matrix(class_labels)
    num_steps = 10
    best_stump = {}
    best_class_est = np.matrix()
    min_error = np.inf
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min)/num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in ["lt", "gt"]:
                threshold_val = range_min + j * step_size
                predicted_vec = stump_classify()
