import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_dataset():
    data_mat = []
    label_mat = []
    with open("testSet.txt", "r") as fr:
        for line in fr.readlines():
            line = line.strip().split('\t')
            data_mat.append([1.0, float(line[0]), float(line[1])])
            label_mat.append(float(line[2]))
    return np.array(data_mat), np.array(label_mat)


def sigmoid(inx):
    return 1.0/(1 + np.exp(-inx))


def grad_ascent(data_mat, class_labels1):
    data_mat = np.matrix(data_mat)
    class_labels = np.matrix(class_labels1).transpose()
    m, n = data_mat.shape
    alpha = 0.01
    maxiter = 500
    weights = np.ones((n, 1))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 1], data_mat[:, 2], 15 * (class_labels1 + 1), 30 * (class_labels1 + 1))
    # x = np.arange(-3, 3, 0.1)

    for i in range(maxiter):
        h = sigmoid(data_mat * weights)
        error = class_labels - h

        # if i % 20 == 0:
        #     y = (-float(weights[0]) - x * float(weights[1])) / float(weights[2])
        #     ax.plot(x, y)

        weights = weights + alpha * data_mat.transpose() * error

    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # plt.show()

    return weights


def sto_grad_ascent0(data_mat, class_labels):
    m, n = np.shape(data_mat)

    max_iter = 200
    plot_weights = []
    weights = np.ones(n)
    plot_weights.append(weights)
    for j in range(max_iter):
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            h = sigmoid(sum(data_mat[i] * weights))
            error = class_labels[i] - h
            weights = weights + alpha * data_mat[i] * error
            plot_weights.append(weights)

    fig = plt.figure()
    plot_weights = np.array(plot_weights)
    ax1 = fig.add_subplot(311)
    ax1.plot(range(max_iter * m+1), plot_weights[:, 0])
    ax1 = fig.add_subplot(312)
    ax1.plot(range(max_iter* m+1), plot_weights[:, 1])
    ax1 = fig.add_subplot(313)
    ax1.plot(range(max_iter * m+ 1), plot_weights[:, 2])
    return weights



def plot_best_fit(weights):
    data_mat, label_mat = load_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 1], data_mat[:, 2], 15 * (label_mat+1), 30 * (label_mat+1))
    x = np.arange(-3, 3, 0.1)
    y = (-float(weights[0]) - x * float(weights[1]))/float(weights[2])
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    import logregre
    import matplotlib.pyplot as plt
    data_arr, label_mat = logregre.load_dataset()
    # plt.scatter(data_arr[:, 1], data_arr[:, 2], 15 * (label_mat+1), 30 * (label_mat+1))
    # plt.show()
    # weights = logregre.grad_ascent(data_arr, label_mat)
    # logregre.plot_best_fit(weights)

    weights = logregre.sto_grad_ascent0(data_arr, label_mat)
    logregre.plot_best_fit(weights)
