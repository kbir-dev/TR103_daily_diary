# Day 22: Cluster Analysis

## Topics
- Unsupervised learning
- K-Means clustering
- Hierarchical clustering
- Cluster evaluation

## Journal

Practiced clustering techniques to find natural groupings in data. Used K-Means and hierarchical clustering. Evaluated with silhouette score.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sample data
X, _ = make_blobs(n_samples=300, centers=4)

# K-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
print(f"Silhouette Score: {silhouette_score(X, kmeans.labels_):.4f}")

# Visualize clusters
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='X', s=200, c='red')
```

## Reflections
Clustering reveals hidden patterns without labels. K-Means is efficient but sensitive to initialization. Elbow method and silhouette score help determine optimal cluster count.

## Resources
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
