# Practical 8: DBSCAN Clustering & Noise Detection (-1)
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels) - {-1})
n_noise = list(labels).count(-1)

print(f"Number of Clusters: {n_clusters}")
print(f"Number of Outliers/Noise (-1): {n_noise}")
if n_clusters > 1:
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title(f"DBSCAN: {n_clusters} Clusters, {n_noise} Outliers")
plt.show()
