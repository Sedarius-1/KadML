import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids (k,data):

    n_dims = data.shape[1]
    centroid_min = data.min().min()
    centroid_max = data.max().max()
    centroids=[]

    for centroid in range(k):
        centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
        centroids.append(centroid)

    centroids = pd.DataFrame(centroids, columns = data.columns)
    return centroids


def calculate_errors(a,b):
    error = np.square(np.sum((a-b)**2))
    return error


def assign_centroid(data, centroids):
    n_observations = data.shape[0]
    centroid_assign = []
    centroid_errors = []
    k = centroids.shape[0]

    for observation in range(n_observations):
        errors = np.array([])
        for centroid in range(k):
            error = calculate_errors(centroids.iloc[centroid, :2], data.iloc[observation, :2])
            errors = np.append(errors, error)

        closest_centroid = np.where(errors == np.amin(errors))[0].tolist()[0]
        centroid_error = np.amin(errors)
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    return centroid_assign, centroid_errors


def KNN_from_scratch(data, k):
    centroids = initialize_centroids(k, data)
    error = []
    compr = True
    i = 0

    while(compr):
        data['centroid'], iter_error = assign_centroid(data, centroids)
        error.append(sum(iter_error))
        centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)

        if (len(error) < 2):
            compr = True
        else:
            if(round(error[i],3) != (error[i-1],3)):
                compr = True
            else:
                compr = False
        i = i + 1

        data['centroid'], iter_error = assign_centroid(data, centroids)
        centroids = data.groupby('centroid').agg('mean').reset_index(drop=True)
        return (data['centroid'], iter_error, centroids)

np.random.seed(15)

data = [[np.random.uniform(0.0, 3.0), np.random.uniform(0.0, 3.0)] for _ in range(0,33)]
for _ in range(33,66):
    data.append([np.random.uniform(5.0, 7.0), np.random.uniform(5.0, 7.0)])
for _ in range(66,99):
    data.append([np.random.uniform(1.0, 4.0), np.random.uniform(8.0, 10.0)])


df = pd.DataFrame(data, columns=['Length', 'Width'])
# centroids = initialize_centroids(3, df)
#
# errors = np.array([])
# for centroid in range(centroids.shape[0]):
#     error = calculate_errors(centroids.iloc[centroid,:2], df.iloc[0,:2])
#     errors= np.append(errors, error)
#
# print(errors)
#
# df['centroid'], df['error'] = assign_centroid(df.iloc[:,:2] ,centroids)
#
# error_sum = df['error'].sum()
# df_columns=['Length', 'Width']
#

#
# plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o', c = df['centroid'].apply(lambda x: colors[x]), alpha = 0.5)
# plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=300,
#            c = centroids.index.map(lambda x: colors[x]))
# plt.show()
#
# centroids = df.groupby('centroid').agg('mean').loc[:,df_columns].reset_index(drop = True)
#
# print(centroids)
#
# colors = {0:'red', 1:'blue', 2:'green'}
#
# plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o', c = df['centroid'].apply(lambda x: colors[x]), alpha = 0.5)
# plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=300,
#            c = centroids.index.map(lambda x: colors[x]))
# plt.show()

colors = {0:'red', 1:'blue', 2:'green'}
df['centroid'], _, centroids =  KNN_from_scratch(df,3)
df['centroid'].head()

plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o', c = df['centroid'].apply(lambda x: colors[x]), alpha = 0.5)
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=300,
           c = centroids.index.map(lambda x: colors[x]))
plt.show()