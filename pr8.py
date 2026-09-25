# ==============================================================================
# Unsupervised Learning Technique Practical Manual - Practical No 8
# Aim: To implement the DBSCAN (Density-Based Spatial Clustering of Applications
#      with Noise) algorithm using Python and Scikit-learn.
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------------------
# 1. Create Sample Dataset
# ------------------------------------------------------------------------------
X, y = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.6,
    random_state=42
)


# ------------------------------------------------------------------------------
# 2. Standardise the Data
# ------------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------------------------
# 3. Apply DBSCAN
# ------------------------------------------------------------------------------
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)
labels = dbscan.fit_predict(X_scaled)


# ------------------------------------------------------------------------------
# 4. Add Cluster Labels to DataFrame
# ------------------------------------------------------------------------------
df = pd.DataFrame(
    X,
    columns=["Feature1", "Feature2"]
)
df["Cluster"] = labels
print("First 5 records with Cluster Labels:\n", df.head())


# ------------------------------------------------------------------------------
# 5. Visualize DBSCAN Clusters
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis',
    s=50
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")
plt.show()


# ------------------------------------------------------------------------------
# 6. Count the Number of Clusters & Noise Points
# ------------------------------------------------------------------------------
unique_labels = set(labels)
n_clusters = len(unique_labels - {-1})
n_noise = list(labels).count(-1)

print(f"\nNumber of Clusters: {n_clusters}")
print(f"Number of Noise Points (-1): {n_noise}")


# ------------------------------------------------------------------------------
# 7. Calculate Silhouette Score (if clusters > 1)
# ------------------------------------------------------------------------------
if n_clusters > 1:
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {score:.4f}")
