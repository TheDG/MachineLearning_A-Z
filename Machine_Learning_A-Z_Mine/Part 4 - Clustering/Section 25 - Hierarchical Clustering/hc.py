# Hierarchical Clustering

# %% codecell
# importing libaries
import data_preprocessing_template as preprocessed_data
import sys
import os
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
if '__file__' not in globals():
    sys.path.append(os.getcwd(
    ) + '/Machine_Learning_A-Z_Mine/Part 4 - Clustering/Section 25 - Hierarchical Clustering')


# %% codecell
# preprocess data
x = preprocessed_data.preprocess_data()


# %% codecell
# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# %% codecell
# Fitting Hierarchical Clustering to the dataset
hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)


# %% codecell
# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(x[y_hc == 5, 0], x[y_hc == 5, 1], s = 100, c = 'yellow', label = 'Cluster 6')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
