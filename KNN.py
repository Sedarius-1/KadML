# from sklearn import datasets
import numpy as np
# import random
# import scipy.stats as ss
# import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier


def min_dist(a, b, p=1):
    dim = len(a)
    distance = .0

    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p
    distance = distance ** (1 / p)
    return distance


def knn_predict(x_training, x_testing, y_training, k_param, p_param):
    y_knn_test = []

    for test_point_index in range(0, x_testing.shape[0]):
        distances = []

        for train_point_index in range(0, x_training.shape[0]):
            distance = min_dist(x_testing.iloc[test_point_index], x_training.iloc[train_point_index], p=p_param)
            distances.append(distance)

        df_dists = pd.DataFrame(data=distances, columns=['dist'],
                                index=y_training.index)

        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k_param]

        counter = Counter(y_training[df_nn.index])

        prediction = counter.most_common()[0][0]

        y_knn_test.append(prediction)

    return y_knn_test


def modify_error_matrix(y_test, y_hat_test, index, error_matrix):
    if y_test[index] == 0:
        if y_hat_test[index] == 0:
            error_matrix[0] += 1
        elif y_hat_test[index] == 1:
            error_matrix[1] += 1
        elif y_hat_test[index] == 2:
            error_matrix[2] += 1
    if y_test[index] == 1:
        if y_hat_test[index] == 0:
            error_matrix[3] += 1
        elif y_hat_test[index] == 1:
            error_matrix[4] += 1
        elif y_hat_test[index] == 2:
            error_matrix[5] += 1
    if y_test[index] == 2:
        if y_hat_test[index] == 0:
            error_matrix[6] += 1
        elif y_hat_test[index] == 1:
            error_matrix[7] += 1
        elif y_hat_test[index] == 2:
            error_matrix[8] += 1


data_train_unnormalized = pd.read_csv('data_train.csv', header=None)
data_test_unnormalized = pd.read_csv('data_test.csv', header=None)

data_train = data_train_unnormalized.copy().iloc[:, :-1]
data_test = data_test_unnormalized.copy().iloc[:, :-1]

for column in data_train.columns:
    data_train[column] = \
        (data_train[column] - data_train[column].min()) / (data_train[column].max() - data_train[column].min())

for column in data_test.columns:
    data_test[column] = \
        (data_test[column] - data_test[column].min()) / (data_test[column].max() - data_test[column].min())

data_train = data_train.sample(frac=1)
data_test = data_test.sample(frac=1)

x_train = data_train
x_test = data_test
x_trainsets = []
x_testsets = []

for i1 in range(0, 4):
    for i2 in range(i1 + 1, 4):
        trainset = data_train.iloc[:, [i1, i2]].copy()
        trainset.columns = ['0', '1']
        x_trainsets.append(trainset)
        testset = data_test.iloc[:, [i1, i2]].copy()
        testset.columns = ['0', '1']
        x_testsets.append(testset)

y_train = data_train_unnormalized.iloc[:, -1]
y_test = data_test_unnormalized.iloc[:, -1]

error_matrices = []
accuracies = []
sklearn_acc = []
top_accuracies = []
paired_error_matrices = [[] for _ in range(0, 6)]
paired_accuracies = [[] for _ in range(0, 6)]
paired_sklearn_acc = [[] for _ in range(0, 6)]
#
for curr_k in range(1, 16):

    error_matrix = np.zeros(9)
    y_hat_test = knn_predict(x_train, x_test, y_train, k_param=curr_k, p_param=2)
    for index in range(len(y_hat_test)):
        modify_error_matrix(y_test.values, y_hat_test, index, error_matrix)
    print(f"My KNN Accuracy for k = {curr_k}: {accuracy_score(y_test, y_hat_test)}")
    accuracies.append([curr_k, accuracy_score(y_test, y_hat_test)])
    error_matrices.append(error_matrix)

    clf = KNeighborsClassifier(n_neighbors=curr_k, p=2)
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    sklearn_acc.append([curr_k, accuracy_score(y_test, y_pred_test)])
    print(f"Sklearn KNN Accuracy: {accuracy_score(y_test, y_pred_test)}")

    for i in range(0, 6):
        error_matrix = np.zeros(9)
        y_hat_test = knn_predict(x_trainsets[i], x_testsets[i], y_train, k_param=curr_k, p_param=2)
        for index in range(len(y_hat_test)):
            modify_error_matrix(y_test.values, y_hat_test, index, error_matrix)
        print(f"Our KNN Accuracy for pair {i} and k = {curr_k}: {accuracy_score(y_test, y_hat_test)}")
        paired_error_matrices[i].append(error_matrix)
        paired_accuracies[i].append([curr_k, accuracy_score(y_test, y_hat_test)])
        clf = KNeighborsClassifier(n_neighbors=curr_k, p=2)
        clf.fit(x_trainsets[i], y_train)
        y_pred_test = clf.predict(x_testsets[i])
        paired_sklearn_acc[i].append([curr_k, accuracy_score(y_test, y_pred_test)])
        print(f"Sklearn KNN Accuracy: {accuracy_score(y_test, y_pred_test)}")


pairs = {0: "Sepal Length and Sepal Width", 1: "Sepal Length and Petal Length", 2: "Sepal Length and Petal Width",
         3: "Sepal Width and Petal Length", 4: "Sepal Width and Petal Width", 5: "Petal Length and Petal Width"}


for accuracy in paired_accuracies:
    max_element = accuracy[0]
    for element in accuracy:
        if element[1] > max_element[1]:
            max_element = element
    top_accuracies.append(max_element)

max_element = accuracies[0]
for element in accuracies:
    if element[1] >= max_element[1]:
        max_element = element

f = open("output/raport.txt", "w")
general = f"For General Classification, best accuracy was achieved for k = {max_element[0]}, equal to {max_element[1]},\n " \
           f" with error matrix:{error_matrices[max_element[0]]}\n"
f.write(general)
print(general)

for index in range(0, 6):
    appended = top_accuracies[index]
    matrix = paired_error_matrices[index][appended[0] - 1]
    appended.append(matrix.tolist())
    part = f"\nFor pair {pairs[index]}, best accuracy was achieved for k = {appended[0]}, equal to {appended[1]},\n" \
          f" with error matrix:{appended[2]}"
    f.write(part)
    print(part)

plt.title("ALL 4 PARAMETERS")
plt.plot(*zip(*accuracies))
plt.plot(*zip(*sklearn_acc), color="purple")
plt.savefig("output/GeneralClassification.png")
plt.show()

for index in range(0, 6):
    plt.title(pairs[index])
    x, y = zip(*paired_accuracies[index])
    plt.plot(x, y)
    x_sk, y_sk = zip(*paired_sklearn_acc[index])
    plt.plot(x_sk, y_sk, color="purple")
    plt.savefig("output/" + pairs[index] + ".png")
    plt.show()

f.close()
