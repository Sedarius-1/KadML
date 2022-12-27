from sklearn import datasets
import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def min_dist(a, b, p=1):
    dim = len(a)
    distance = .0

    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p
    distance = distance ** (1 / p)
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):

    y_hat_test = []

    for test_point_index in range(0,X_test.shape[0]):
        distances = []

        for train_point_index in range(0,X_train.shape[0]):
            distance = min_dist(X_test.iloc[test_point_index], X_train.iloc[train_point_index], p=p)
            distances.append(distance)

        df_dists = pd.DataFrame(data=distances, columns=['dist'],
                                index=y_train.index)

        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        counter = Counter(y_train[df_nn.index])

        prediction = counter.most_common()[0][0]

        y_hat_test.append(prediction)

    return y_hat_test


data_train = pd.read_csv('data_train.csv', header=None)

data_test = pd.read_csv('data_test.csv', header=None)

data_train = data_train.sample(frac = 1)
data_test = data_test.sample(frac = 1)

x_train = data_train.iloc[: , :-1]
y_train = data_train.iloc[: , -1]
x_test = data_test.iloc[: , :-1]
y_test = data_test.iloc[: , -1]

for curr_k in range (80,95):
    y_hat_test = knn_predict(x_train, x_test, y_train, y_test, k=curr_k, p=2)
    print(y_hat_test)
    print(f"My KNN Accuracy for k = {curr_k}: {accuracy_score(y_test, y_hat_test)}")
    clf = KNeighborsClassifier(n_neighbors=curr_k, p=2)
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    print(y_pred_test)
    print(f"Sklearn KNN Accuracy: {accuracy_score(y_test, y_pred_test)}")

