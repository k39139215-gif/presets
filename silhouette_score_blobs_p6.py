# Practical 6: Silhouette Score & Sample Scores on Synthetic Data
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print(f"Overall Silhouette Score: {silhouette_score(X, labels):.3f}")
sample_scores = silhouette_samples(X, labels)
print("First 5 Sample Scores:", sample_scores[:5])
