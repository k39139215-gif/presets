# ==============================================================================
# PRACTICAL 8: DBSCAN CLUSTERING & OUTLIER DETECTION
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. Dataset Generation & Standardization
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 2. Fit DBSCAN (eps=0.3, min_samples=5)
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X_scaled)

# 3. Analyze Clusters and Noise (-1 represents outliers)
unique_labels = set(labels)
n_clusters = len(unique_labels - {-1})
n_noise = list(labels).count(-1)

print(f"Number of Clusters Formed: {n_clusters}")
print(f"Number of Noise Points (-1): {n_noise}")

if n_clusters > 1:
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")

# 4. Scatter Plot of DBSCAN
plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=40)
plt.title(f"DBSCAN: {n_clusters} Clusters, {n_noise} Outliers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
