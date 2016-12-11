import numpy as np


def load_simple_data():
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


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
    label_mat = np.matrix(class_labels).T
    num_steps = 10
    best_stump = {}
    best_class_est = np.matrix(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min)/num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in ["lt", "gt"]:
                threshold_val = range_min + j * step_size
                predicted_vec = stump_classify(data_mat, i, threshold_val, inequal)
                err_arr = np.matrix(np.ones((m, 1)))
                err_arr[predicted_vec == label_mat] = 0
                weighted_error = weight.T*err_arr
                # print "split: dim %d, threshold %.2f, threshold inequal: %s, the weighted error is %.3f" % \
                #       (i, threshold_val, inequal, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vec.copy()
                    best_stump["dim"] = i
                    best_stump["threshold"] = threshold_val
                    best_stump["inequality"] = inequal
    return best_stump, min_error, best_class_est


def adaboost_train(data_arr, class_labels, num_iter=40):
    weak_class_arr = []
    m = data_arr.shape[0]
    D = np.matrix(np.ones((m, 1))/m)
    agg_class_est = np.matrix(np.zeros((m, 1)))
    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
<<<<<<< HEAD
        print "D:", D.T
        alpha = float(0.5 * np.log((1.0-error)/max(error,1e-16)))
        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)
        print "class_est:", class_est.T
        agg_class_est += alpha * class_est
        print "aggr_class_est:", agg_class_est.T
=======
        # print "D:", D.T
        alpha = float(0.5 * np.log((1.0-error)/error))
        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)
        # print "class_est:", class_est.T
        agg_class_est += alpha * class_est
        # print "aggr_class_est:", agg_class_est.T
>>>>>>> 68d745d9f4a5c0b75e4d74d7d47153b544aab2b9
        # update the weight
        expon = np.exp(np.multiply(-alpha*class_est, np.mat(class_labels).T))
        D = np.multiply(D, expon)
        D = D/D.sum()

        agg_error = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_error.sum()/m
<<<<<<< HEAD
        print "total error: ", error_rate, "\n"
=======
        # print "total error: ", error_rate, "\n"
>>>>>>> 68d745d9f4a5c0b75e4d74d7d47153b544aab2b9
        if error_rate == 0:
            break
    return weak_class_arr


<<<<<<< HEAD
if __name__ == '__main__':
    import adaboost
    data_mat, class_labels = adaboost.load_simple_data()
    a = adaboost.adaboost_train(data_mat, class_labels, 9)
    print a
=======
def ada_classify(datto_class, classifier_arr):
    data_mat = np.matrix(datto_class)
    m = data_mat.shape[0]
    agg_class_est = np.zeros((m, 1))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, classifier_arr[i]["dim"], classifier_arr[i]["threshold"],
                                        classifier_arr[i]["inequality"])
        agg_class_est += classifier_arr[i]["alpha"] * class_est
        # print agg_class_est
    # print np.sign(agg_class_est)
    return np.sign(agg_class_est)

def load_dataset(filename="horseColicTraining2.txt"):
    with open(filename, "r") as fr:
        num_features = len(fr.readline().strip().split("\t"))-1
    data_mat = []; label_mat = []
    with open(filename, "r") as fr:
        for line in fr.readlines():
            line_arr = []
            curline = line.strip().split("\t")
            for i in range(num_features):
                line_arr.append(float(curline[i]))
            data_mat.append(line_arr)
            label_mat.append(float(curline[-1]))
    return np.matrix(data_mat), np.matrix(label_mat)





if __name__ == '__main__':
    import adaboost
    # data_mat, labels_arr = adaboost.load_simple_data()
    # classifier_arr = adaboost.adaboost_train(data_mat, labels_arr, 9)
    # ada_classify([[0, 0], [5, 5]], classifier_arr)
    data_arr, labels_arr = load_dataset()
    test_arr, test_labels_arr = load_dataset("horseColicTest2.txt")
    error_rate = []
    # for i in [1, 10, 50, 100, 500, 1000]:
    classifier_arr = adaboost.adaboost_train(data_arr, labels_arr, 10)
    prediction10 = ada_classify(test_arr, classifier_arr)
    error_mat = np.mat(np.zeros((67, 1)))
    error_mat[prediction10 != np.mat(test_labels_arr).T] = 1.0
    error_rate.append(error_mat.sum())
    print error_rate
>>>>>>> 68d745d9f4a5c0b75e4d74d7d47153b544aab2b9
