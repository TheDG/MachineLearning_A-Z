# Intro
Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.

# K-Means
Non deterministic —> Final clusters depend on random initialization, which can result with non optimal clusters. Solution to this is K-Means++ algorithm
### Algorithm
    1. Choose K Number of clusters
    2. Select at random K points —> centroids. Not necessarily part of dataset
    3. Assign each datapoint to the closet centroid
      * Distance can be any function
    4. Compute and place new centroid for each cluster
    5. Reassign each datapoint to new closet centroid
      * If reassignment took place, go back to step 4. Otherwise, END

### Choosing the right # of clusters

- Use Within Cluster Sum of Squares [WCSS] as a metric to decided.
- Minimize squared distance from points to their centroids (for each cluster).
- WCSS will converge to 0 when # of clusters = # of data points. —> Elbow method
- Choose point where rate of change(derivative) has a drastic change —> starts decreasing much slower

# Hierachical Clustering
Agglomerative: Bottom up approach

Divise: Top down approach

### Agglomerative Algorithm
    1. Make each data point into a single cluster —> form n clusters
    2. Take the closet 2 clusters and join them into 1 cluster —> forms n-1 clusters
        * How to measure distance between clusters? It’s not just points?
        * Using Euclidean Distance —>
            * Distance between closest points, furthest points, average distance, distance from centroids?
    3. Repeat step 2 until you only have 1 cluster (or wanted amount of clusters)

### Choosing the right # of clusters
 Use  **Dendogram** --> Diagram that shows the hierarchical relationship between objects

    1. Build Dendogram
    2. Find longest distance (vertical line that is not cut by any horizontal extended line)
    3. Draw cutoff line, each vertical line that is intersected, would be a cluster

# Clustering Model Pro - Cons

### K-Means
* Pros: Simple to understand, easily adaptable, works well on small or large datasets, fast, efficient and performant
* Cons: Need to choose the number of clusters

### Hierarchical Clustering
* Pros: The optimal number of clusters can be obtained by the model itself, practical visualisation with the dendrogram
* Cons: Not appropriate for large datasets
