# Day 27: Cluster Analysis Practice

## Topics
- Customer segmentation
- Dimensionality reduction
- Cluster profiling
- Business applications

## Journal

Segmented customers using RFM analysis (Recency, Frequency, Monetary) with K-Means clustering. Used PCA for visualization.

```python
from sklearn.decomposition import PCA

# RFM features
rfm = df[['Recency', 'Frequency', 'Monetary']]

# K-Means clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(rfm)

# PCA visualization
pca = PCA(n_components=2)
rfm_2d = pca.fit_transform(rfm)
plt.scatter(rfm_2d[:,0], rfm_2d[:,1], c=clusters)
```

## Reflections
Effective segmentation requires meaningful features. Cluster profiling reveals distinct customer behaviors. Business applications include targeted marketing and product development.

## Resources
- [RFM Analysis Guide](https://www.putler.com/rfm-analysis/)
