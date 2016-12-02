import numpy as np
import re

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


def bag_of_words_to_vec(vocab_list, input_set):
    vocab_list = list(vocab_list)
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
        else:
            p0_num += train_matrix[i]
    p1_vect = p1_num / sum(p1_num)
    p0_vect = p0_num / sum(p0_num)
    return np.log(p0_vect), np.log(p1_vect), p_abusive


def classify_nb(vec2calssify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2calssify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2calssify * p0_vec) + np.log(1-p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_of_posts, list_classes = load_dataset()
    my_vocab_list = create_vocab_list(list_of_posts)
    train_mat = []
    for post_in_doc in list_of_posts:
        train_mat.append(bag_of_words_to_vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = train_nb0(train_mat, list_classes)
    # test_entry = list("love my dalmation".split(" "))
    test_entry = list("stupid garbage".split(" "))
    this_doc = np.array(bag_of_words_to_vec(my_vocab_list, test_entry))
    print test_entry," classified as: ", classify_nb(this_doc, p0_v, p1_v, p_ab)


def test_parse(big_string):
    list_of_tokens = re.split(r"\W*", big_string)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


def spam_test():
    doc_list = []; class_list = []; full_text = []
    for i in range(1, 26):
        word_list = test_parse(open("email/ham/%d.txt" % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
        word_list = test_parse(open("email/spam/%d.txt" % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

    vocab_list = set(full_text)
    training_set = range(50); test_set = []
    for i in range(int(15)):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    training_mat = []; training_classes = []
    for doc_index in training_set:
        training_mat.append(set_of_words_to_vec(vocab_list, doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb0(np.array(training_mat), np.array(training_classes))
    error_count = 0
    for doc_index in test_set:
        test_mat = set_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_nb(np.array(test_mat), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    error_rate = error_count / float(len(test_set))
    # print "the error rate is: ", error_rate
    return error_rate


def cal_most_freq(vocalb_list, full_text):
    import operator
    freq_dict = {}
    for token in vocalb_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list = []; class_list = []; full_text = []
    min_len = min(len(feed1["entries"]), len(feed0["entries"]))
    for i in range(min_len):
        word_list = test_parse(feed1["entries"][i]["summary"])
        doc_list.append(word_list)
        class_list.append(1)
        full_text.extend(word_list)
        word_list = test_parse(feed0["entries"][i]["summary"])
        doc_list.append(word_list)
        class_list.append(0)
        full_text.extend(word_list)

    vocab_list = set(full_text)
    top30_words = cal_most_freq(vocab_list, full_text)
    # vocab_list = vocab_list - set(top30_words)
    stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","aren\'t","as","at",
                  "be","because","been","before","being","below","between","both","but","by","can\'t","cannot","could",
                  "couldn\'t","did","didn\'t","do","does","doesn\'t","doing","don\'t","down","during","each","few","for"
        ,"from","further","had","hadn\'t","has","hasn\'t","have","haven\'t","having","he","he\'d","he\'ll","he\'s","her"
        ,"here","here\'s","hers","herself","him","himself","his","how","how\'s","i","i\'d","i\'ll","i\'m","i\'ve","if",
                  "in","into","is","isn\'t","it","it\'s","its","itself","let\'s","me","more","most","mustn\'t","my",
                  "myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours\tourselves",
                  "out","over","own","same","shan\'t","she","she\'d","she\'ll","she\'s","should","shouldn\'t","so","some"
        ,"such","than","that","that\'s","the","their","theirs","them","themselves","then","there","there\'s","these","they","they\'d","they\'ll","they\'re","they\'ve","this","those","through","to","too","under","until","up","very","was","wasn\'t","we","we\'d","we\'ll","we\'re","we\'ve","were","weren\'t","what","what\'s","when","when\'s","where","where\'s","which","while","who","who\'s","whom","why","why\'s","with","won\'t","would","wouldn\'t","you","you\'d","you\'ll","you\'re","you\'ve","your","yours","yourself","yourselves"]

    vocab_list = vocab_list - set(stop_words)
    training_set = range(2 * min_len); test_set = []
    for i in range(int(2 * min_len * 0.3)):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    training_mat = []; training_classes = []
    for doc_index in training_set:
        training_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb0(np.array(training_mat), np.array(training_classes))
    error_count = 0
    for doc_index in test_set:
        test_mat = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_nb(np.array(test_mat), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    error_rate = error_count / float(len(test_set))
    # print "the error rate is: ", error_rate
    return error_rate

if __name__ == "__main__":
    import bayes
    import feedparser
    # error_rate = []
    # for i in range(100):
    #     error_rate.append(bayes.spam_test())
    # print "the mean error rate is: ", np.array(error_rate).mean()

    ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")

    error_rate = []
    for i in range(100):
        error_rate.append(bayes.local_words(ny, sf))
    print "the mean error rate is: ", np.array(error_rate).mean()

