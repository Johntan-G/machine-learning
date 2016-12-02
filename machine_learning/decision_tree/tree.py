import math
import operator
import matplotlib.pyplot as plt

def create_dataset():
    dataset = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataset, labels


def cal_shannon_ent(dataset):
    num_entries = len(dataset)
    label_count = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        label_count[current_label] = label_count.get(current_label, 0) + 1
    shannon_ent = 0.0
    for key in label_count:
        prob = label_count[key] / float(num_entries)
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduce_feature_vec = feature_vec[:axis]
            reduce_feature_vec.extend(feature_vec[axis+1:])
            ret_dataset.append(reduce_feature_vec)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    num_feature = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        feature_list = [example[i] for example in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * cal_shannon_ent(sub_dataset)
        if base_entropy - new_entropy > best_info_gain:
            best_info_gain = base_entropy - new_entropy
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_sort = sorted(class_count.iteritems(), \
                               key=operator.itemgetter(1), reverse=True)
    return sorted_class_sort[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:# this one is label
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in dataset]
    unique_vals = set(feature_values)
    for value in unique_vals:
        sublabels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_dataset\
                                (dataset, best_feature, value), sublabels)
    return my_tree


def classify(input_tree, feature_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == "dict":
                class_label = classify(second_dict[key], feature_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

if __name__ == "main":
    pass