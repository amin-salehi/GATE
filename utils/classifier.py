from __future__ import print_function
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import numpy as np


class Classifier(object):

    def __init__(self, vectors):
        self.embeddings = vectors

    def __call__(self, train_index, test_index, val_index, Y, seed=0):

        numpy.random.seed(seed)

        # Y_train = []
        # X_test = []
        # for node, vec in enumerate(Y):
        #     for class_, value in enumerate(vec):
        #         if value == 1:
        #             Y_train.append(class_)

        #Y = np.array(labels)

        averages = ["micro", "macro"] #, "samples", , "weighted"
        f1s = {}

        Y = np.argmax(Y, -1)


        X_train = [self.embeddings[x] for x in train_index]
        Y_train = [Y[x] for x in train_index]
        X_test = [self.embeddings[x] for x in test_index]
        Y_test = [Y[x] for x in test_index]

        clf = LogisticRegression()
        clf.fit(X_train, Y_train)
        Y_  = clf.predict(X_test)


        for average in averages:
            f1s[average]= f1_score(Y_test, Y_, average=average)

        #print(f1s)


        return f1s



def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors

def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(int(vec[0]))
        Y.append([int(y) for y in vec[1:]])
    fin.close()
    return X, Y