import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = list("AABB")
    return group, labels


def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_distance = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distance[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filepath="datingTestSet.txt"):
    with open(filepath, mode='r') as f:
        array_of_lines = f.readlines()
        number_of_lines = len(array_of_lines)
        return_mat = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_of_lines:
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]
            class_label_vector.append(list_from_line[-1])
            index += 1
    replace_dict = {"largeDoses": 3, "smallDoses": 2, "didntLike": 1}
    class_label_vector = map(lambda x: replace_dict[x], class_label_vector)
    return return_mat, class_label_vector


def autonorm(dataset):
    # min_vals = dataset.min(axis=0)
    # max_vals = dataset.max(axis=0)
    # rangs = max_vals - min_vals
    # m = dataset.shape[0]
    # norm_dataset = dataset - np.tile(min_vals, (m, 1))
    # norm_dataset = norm_dataset / rangs
    # another method to normalize
    import sklearn.preprocessing
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler = scaler.fit(dataset)
    norm_dataset = scaler.transform(dataset)
    rangs = scaler.data_range_
    scaler.data_range_
    min_vals = scaler.data_min_
    return norm_dataset, scaler


def dating_class_test(k=3):
    ho_ration = 0.1
    dating_data_mat, dating_labels = file2matrix()
    norm_dataset, ranges, min_vals = autonorm(dating_data_mat)
    m = dating_data_mat.shape[0]
    num_test_vecs = int(m * ho_ration)
    error_count = 0
    for i in range(num_test_vecs):
        classify_result = classify0(norm_dataset[i, :], norm_dataset[num_test_vecs:m, :],
                                    dating_labels[num_test_vecs:m], k)
        # print "The classifier came back with: %d, the real answer is: %d" % (classify_result, dating_labels[i])
        if classify_result != dating_labels[i]:
            error_count += 1
    print "the total error rate is: %f, where k equals %d" % (error_count / float(num_test_vecs), k)


def classify_person():
    result_list = ["not at all", "in small does", "in large does"]
    percent_tats = float(raw_input("percentage of time spent palying video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per years?"))
    icecream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix()
    norm_dataset, scaler = autonorm(dating_data_mat)
    classify_result = classify0(scaler.transform([percent_tats, ff_miles, icecream]), dating_data_mat, dating_labels, 4)
    print "you will probably like this person", result_list[classify_result-1]


def img2vector(filepath):
    return_vector = np.zeros((1, 1024))
    with open(filepath) as f:
        for i in range(32):
            line = f.readline()
            line = line.strip()
            line = list(line)
            line = map(int, line)
            return_vector[0, 32*i:32*(i+1)] = line
    return np.squeeze(return_vector)


def handwriting_class_test(k=3):
    import os
    hwlabels = []
    training_file_list = os.listdir("digits/trainingDigits")
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        filenamestr = training_file_list[i]
        filestr = filenamestr.split(".")[0]
        class_number = filestr.split("_")[0]
        hwlabels.append(class_number)
        training_mat[i, :] = img2vector("digits/trainingDigits/" + filenamestr)

    test_file_list = os.listdir("digits/testDigits")
    n = len(test_file_list)
    error_count = 0.0
    for i in range(n):
        filenamestr = test_file_list[i]
        class_number = filenamestr.split(".txt")[0].split("_")[0]
        test_mat = img2vector("digits/trainingDigits/" + filenamestr)
        classify_result = classify0(test_mat, training_mat, hwlabels, k)
        if classify_result != class_number:
            error_count += 1
        # print "The classifier came back with: %s, the real answer is: %s" % (classify_result, class_number)
    print "the total error rate is: %f, where k equals %d" % (error_count / float(n), k)


if __name__ == '__main__':
    for k in range(1, 8):
        handwriting_class_test(k)


    # dating_data_mat, dating_labels = kNN.file2matrix("/home/johntan/Documents/PycharmProjects/knn/dataSet/Ch02/datingTestSet.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
    #            15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # plt.show()
