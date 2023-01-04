import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand


def initialize_centroids(data, data_length, K):
    centroids = []
    for i in range(K):
        centroid = data[rand.randint(0, data_length-1)]
        centroids.append(centroid)
    return centroids


def pick_cluster(data, centroids):
    assignments = []

    for point in data:
        distance_list = []

        for centroid in centroids:
            distance = np.linalg.norm(np.array(point) - np.array(centroid))
            distance_list.append(distance)

        closest_cluster = np.argmin(distance_list)
        assignments.append(closest_cluster)

    return assignments


def reassign_centroids(data, data_length, assignments, K):
    reassigned_centroids = []
    for i in range(K):
        clustered_points = []
        for n in range(data_length):
            if assignments[n] == i:
                clustered_points.append(data[n])
        cluster_mean = np.mean(clustered_points, axis=0)
        reassigned_centroids.append(cluster_mean)
    return reassigned_centroids


def wcss(data, data_length, assignments, centroids):
    distance_list = []

    for i in range(data_length):
        centroid = centroids[assignments[i]]
        distance = np.linalg.norm(np.array(data[i]) - np.array(centroid))
        distance_list.append(distance ** 2)

    wcss_total = sum(distance_list)
    return wcss_total


def kmeans(data, data_length, K, iterations=100, tolerance_level=0.01):
    current_iteration = -1
    wcss_list = []
    assignments = []
    centroids = initialize_centroids(data, data_length, K)
    while len(wcss_list) <= 1 or (current_iteration < iterations and np.absolute(wcss_list[current_iteration] - wcss_list[current_iteration - 1])/ wcss_list[current_iteration - 1] >= tolerance_level):
        current_iteration += 1
        assignments = pick_cluster(data, centroids)
        centroids = reassign_centroids(data, data_length, assignments, K)
        wcss_kMeans = wcss(data, data_length, assignments, centroids)
        wcss_list.append(wcss_kMeans)

    return assignments, centroids, wcss_list, current_iteration + 1


columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
iris_data = pd.read_csv('data.csv', names=columns)

values_list = iris_data.drop(columns=['target']).values.tolist()

calculated_kmeans = kmeans(values_list, len(values_list), K=3)

for a1 in range(0,4):
    for a2 in range(a1+1, 4):
        centroids_x = [calculated_kmeans[1][x][a1] for x in range(len(calculated_kmeans[1]))]
        centroids_y = [calculated_kmeans[1][x][a2] for x in range(len(calculated_kmeans[1]))]
        x = iris_data[columns[a1]]
        y = iris_data[columns[a2]]
        assignments = calculated_kmeans[0]
        plt.scatter(x, y, c=assignments)
        plt.plot(centroids_x, centroids_y, c='white', marker='.', linewidth='0.01', markerfacecolor='red',
                 markersize=24)
        plt.title("K-means Visualization")
        plt.xlabel(columns[a1])
        plt.ylabel(columns[a2])
        plt.savefig("output/kMeans/"+columns[a1]+"-"+columns[a2]+".jpg")
        plt.show()
f = open("output/kMeans/raport.txt", "w")
iteration_list = []
wcss_list = []
for k in range (2,11):
    calculated_kmeans = kmeans(values_list, len(values_list), K=k)
    wcss_list.append([k,calculated_kmeans[2][-1]])
    iteration_list.append(calculated_kmeans[3])
    f.write(f"Amount of iterations for k={k}:{calculated_kmeans[3]}, WCSS = {calculated_kmeans[2][-1]}\n")

plt.plot(*zip(*wcss_list))
plt.savefig("output/kMeans/kMeansElbow.jpg")
plt.show()
f.close()

