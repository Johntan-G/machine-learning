import numpy as np


def load_dataset(filename="ex0.txt"):
    num_features = len(open(filename, "r").readline().strip().split("\t")) - 1
    data_mat = []; label_mat = []
    with open(filename, "r") as fr:
        for line in fr.readlines():
            line_arr = []
            line = line.strip().split("\t")
            for i in range(num_features):
                line_arr.append(float(line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(line[-1]))
    return np.mat(data_mat), np.mat(label_mat).T


def standard_regression(data_mat, label_mat):

    xtx = data_mat.T * data_mat
    if np.linalg.det(xtx) == 0:
        print "this matrix is singular, cannot do inverse"
        return None
    else:
        return xtx.I * (data_mat.T * label_mat)


def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    m = x_arr.shape[0]
    w = np.mat(np.eye(m))
    for i in range(m):
        diff_mat = x_mat[i, :] - test_point
        w[i, i] = np.exp(diff_mat*diff_mat.T/(-2*k**2))
    xtx = x_mat.T * (w*x_mat)
    if np.linalg.det(xtx) == 0:
        print "this matrix is singular, cannot do inverse"
        return None
    else:
        return test_point * xtx.I * (x_mat.T*(w*y_mat))


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = test_arr.shape[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i, :], x_arr, y_arr, k)
    return y_hat


def ridge_regression(x_mat, y_mat, lam=0.2):
    xtx = x_mat.T * x_mat + np.eye(x_mat.shape[1]) * lam
    if np.linalg.det(xtx) == 0:
        print "this matrix is singular, connot do inverse"
        return None
    else:
        w = xtx.I * (x_mat.T * y_mat)
        return w


def ridge_test(x_arr, y_arr):
    x_mat = np.matrix(x_arr)
    x_mean = x_mat.mean(axis=0)
    x_var = x_mat.var(axis=0)
    x_mat = (x_mat - x_mean)/x_var
    y_mat = np.matrix(y_arr)
    y_mean = y_mat.mean(axis=0)
    y_mat = y_mat - y_mean
    num_test = 30
    w_mat = np.mat(np.zeros((x_mat.shape[1], num_test)))
    for i in range(num_test):
        w = ridge_regression(x_mat, y_mat, lam=np.exp(i-15))
        w_mat[:, i] = w
    return w_mat.T


def stagewise(x_arr, y_arr, eps=0.01, numiter=100):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    # normalization
    import sklearn.preprocessing
    scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    x_mat = scaler.fit_transform(x_mat)
    y_mat = y_mat - y_mat.mean(0)
    m, n = np.shape(x_mat)
    return_mat = np.zeros((numiter, n))
    w = np.zeros((n, 1)); w_max = w.copy()
    for i in range(numiter):
        print w.T
        lowest_error = np.inf
        for j in range(n):
            for k in [-1, 1]:
                w_test = w.copy()
                w_test[j] += k * eps
                y_test = x_arr * w_test
                rss = ((y_test.A - y_mat.A)**2).sum()
                if rss < lowest_error:
                    lowest_error = rss
                    w_max = w_test
        w = w_max.copy()
        return_mat[i, :] = w.T
    return return_mat




if __name__ == "__main__":
    import regression
    import matplotlib.pyplot as plt
    # data_mat, label_mat = load_dataset()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 1], label_mat)
    # beta = standard_regression(data_mat, label_mat)
    # index = np.argsort(data_mat[:, 1], axis=0).squeeze(1)
    # y_hat = data_mat * beta
    # ax.plot(data_mat[index].squeeze()[:, 1], y_hat[index].squeeze(0), lw=3)
    # # plt.show()
    # # print np.corrcoef(y_hat.T, label_mat.T)
    #
    # m = data_mat.shape[0]
    # y_hat = lwlr_test(data_mat, data_mat, label_mat, 0.01)
    # index = np.argsort(data_mat[:, 1], axis=0).squeeze(1)
    # ax.plot(data_mat[index].squeeze()[:, 1], y_hat[index].squeeze(0), lw=3)
    # plt.show()

    # ridge regression
    abx, aby = regression.load_dataset("abalone.txt")
    w = stagewise(abx, aby, 0.001, 5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w)
    plt.show()


