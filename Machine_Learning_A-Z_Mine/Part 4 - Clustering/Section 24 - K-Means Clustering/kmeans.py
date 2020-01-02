# K-Means Clustering

# %% codecell
# importing libaries
import data_preprocessing_template as preprocessed_data
import sys
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 4 - Clustering/Section 24 - K-Means Clustering')


# %% codecell
# preprocess data
x = preprocessed_data.preprocess_data()

# %% codecell
# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0,
                    max_iter=300, n_init=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %% codecell
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++',
                random_state=0, max_iter=300, n_init=10)
y_kmeans = kmeans.fit_predict(x)


# %% codecell
# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
            s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1],
            s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
