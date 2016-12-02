import numpy as np


def load_dataset():
    posting_ist = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_ist, class_vec


def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return vocab_set


def set_of_words_to_vec(vocab_list, input_set):
    vocab_list = list(vocab_list)
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
    return return_vec


def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / num_train_docs
    p0_num = np.zeros(num_words)
    p1_num = np.zeros(num_words)
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_category[i]
        else:
            p0_num += train_category[i]
    p1_vect = p1_num / sum(p1_num)
    p0_vect = p0_num / sum(p0_num)
    return p0_vect, p1_vect, p_abusive


if __name__ == "__main__":
    import bayes
    list_of_posts, list_classes = bayes.load_dataset()
    my_vocab_list = bayes.create_vocab_list(list_of_posts)
    train_mat = []
    for post_in_doc in list_of_posts:
        train_mat.qppend(set_of_words_to_vec(list_of_posts))

