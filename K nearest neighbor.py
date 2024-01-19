import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

vectorizer = TfidfVectorizer()
training_file = open("hw1_training_data.txt", "r")
test_file = open("hw1 testing data.txt", "r")


def get_results():  # just gets +1 or -1 for test reviews
    signs = []
    f = open("hw1_training_data.txt", "r")
    for results in f:
        signs.append(results[:2])
    return signs


def remove_stopwords(sentence):  # takes review and removes stop words and puts back into dataframe
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    sentences = []
    for w in words:
        if w not in stop_words:
            sentences.append(w)
    sentences = " ".join(sentences)
    return sentences


def set_test_review(testing_file):
    data = []
    for f in testing_file:
        data.append(f)
    dframe = pd.DataFrame(data, columns=['reviews'])
    dframe["reviews"] = dframe["reviews"].apply(lambda x: remove_stopwords(x))
    return dframe


def set_review(train_file):  # sets reviews into a dataframe
    data = []
    for f in train_file:
        line = [f[:2], f[3:]]
        data.append(line)
    dframe = pd.DataFrame(data, columns=['results', 'reviews'])
    dframe["reviews"] = dframe["reviews"].apply(lambda x: remove_stopwords(x))
    return dframe


def tdidf_fitform(dframe):  # returns tdidf from reviews in dataframe
    bag_of_words = vectorizer.fit_transform(dframe["reviews"].values)
    return bag_of_words


def tdidf_transform(dframe):  # returns tdidf from reviews in dataframe
    bag_of_words = vectorizer.transform(dframe["reviews"].values)
    return bag_of_words


def cross_val(dframe, k, arr):
    start = 0
    divisible = 3701
    results = get_results()
    correct = 0
    total = 0
    for i in range(5):  # 5 partitions 3701 each partition size
        training_data = dframe.drop(dframe.index[start:divisible])
        test_data = dframe.iloc[start:divisible]
        test_data = shuffle(test_data)  # randomize data
        training_vector = tdidf_fitform(training_data)
        print(test_data)
        test_vector = tdidf_transform(test_data)
        print("going into knn")
        arr = knn(test_vector, training_vector, k, start, arr)
        start = divisible
        divisible = divisible + divisible
    for i in arr:  # outputs accuracy prediction
        for j in results:
            if i == j:
                correct += 1
            total += 1
    print(correct / total)
    return correct / total


def get_predictions(k_list, i):
    print("going to predict")
    pos_count = 0
    neg_count = 0
    signals = get_results()
    print(i)
    for j in k_list:  # goes through array with indices of biggest values least to greatest
        if signals[j] == "+1" and i <= 18505:
            print("accessing pos review")
            pos_count = pos_count + 1
        elif signals[j] == "-1" and i <= 18505:
            print("accessing neg review")
            neg_count = neg_count + 1
    print("getting vote majority")
    print(pos_count)
    print(neg_count)
    if pos_count > neg_count:
        answer = "+1"
        print("result is", answer)
        return answer
    elif pos_count < neg_count:
        answer = "-1"
        print("result is", answer)
        return answer
    elif pos_count == neg_count:
        answer = "+1"
        return answer


def knn(test, train, k, arr):  # performs knn for each test point for all of training data
    distance = cosine_similarity(test, train)
    count = 0
    f = open("outputs", "w")
    for dist_arr in distance:
        index_k = dist_arr.argsort()[-k:]
        result = get_predictions(index_k, count)
        count += 1
        arr.append(result)
    for lines in arr:
        f.write(lines + "\n")
    f.close()
    return arr


if __name__ == "__main__":
    ks = 9
    result_arr = []
    test_dataframe = set_test_review(test_file)
    train_dataframe = set_review(training_file)
    test_vector = tdidf_fitform(test_dataframe)
    train_vector = tdidf_transform(train_dataframe)
    knn(test_vector, train_vector, ks, result_arr)
    # cross_val(dataframe, ks, result_arr)
