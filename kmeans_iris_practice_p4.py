# Practical 4: Practice Question 1 (Iris Dataset with K=3)
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X, _ = load_iris(return_X_y=True)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

print("Iris K=3 Cluster Labels:\n", kmeans.labels_)
print("\nCluster Centroids:\n", kmeans.cluster_centers_)
