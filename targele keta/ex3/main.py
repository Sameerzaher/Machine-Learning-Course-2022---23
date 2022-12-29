# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# X, y = make_blobs(n_samples=100, centers=5, random_state=101)
# plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
# plt.scatter(X[:, 0], X[:, 1]);
# plt.show()
# Cluster = KMeans(n_clusters=5)
# Cluster.fit(X)
# y_pred = Cluster.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='plasma')
# plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# using the make_blobs dataset
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=5, random_state=101)
# setting the number of training examples
m = X.shape[0]
n = X.shape[1]
n_iter = 50
# computing the initial centroids randomly
K = 5

# creating an empty centroid array
centroids = np.array([]).reshape(n, 0)

# creating 5 random centroids
for k in range(K):
    centroids = np.c_[centroids, X[random.randint(0, m - 1)]]
output = {}

# creating an empty array
euclid = np.array([]).reshape(m, 0)

# finding distance for each centroid
for k in range(K):
    dist = np.sum((X - centroids[:, k]) ** 2, axis=1)
    euclid = np.c_[euclid, dist]

# storing the minimum value we have computed
minimum = np.argmin(euclid, axis=1) + 1
# computing the mean of separated clusters
cent = {}
for k in range(K):
    cent[k + 1] = np.array([]).reshape(2, 0)

# assigning of clusters to points
for k in range(m):
    cent[minimum[k]] = np.c_[cent[minimum[k]], X[k]]
for k in range(K):
    cent[k + 1] = cent[k + 1].T
# computing mean and updating it
for k in range(K):
    centroids[:, k] = np.mean(cent[k + 1], axis=0)

plt.scatter(X[:, 0], X[:, 1])
plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
plt.title('Original Dataset')
for k in range(K):
    plt.scatter(final[k + 1][:, 0], final[k + 1][:, 1])
plt.scatter(centroids[0, :], centroids[1, :], s=300, c='yellow')
plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
plt.show()
