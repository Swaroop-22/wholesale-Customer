# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:58:28 2021

@author: Swaroop Honrao
"""

#Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
data = pd.read_csv('Wholesale customers data.csv')
x = data.iloc[:, [2,3]].values

#Using Elbow method to obtain optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 30)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()

#Training model on dataset
kmeans = KMeans(n_clusters=5, init = 'k-means++', random_state = 30)
y_kmeans = kmeans.fit_predict(x)

#Visualization
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Food')
plt.ylabel('Milk')
plt.legend()
plt.show()
