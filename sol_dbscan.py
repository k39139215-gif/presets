import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. DATA LOADING
# df = pd.read_csv('filename.csv')
# X = df.select_dtypes(include=np.number).dropna().values
df = pd.DataFrame({
    'Feature1': [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    'Feature2': [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
})
X = df.values

# 2. STANDARDIZE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN CLUSTERING
dbscan = DBSCAN(eps=0.5, min_samples=3)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels) - {-1})
n_noise = list(labels).count(-1)

print(f"Total Clusters Formed: {n_clusters}")
print(f"Noise / Outlier Points (-1): {n_noise}")
print("Cluster Labels:\n", labels)

# 4. PLOT DBSCAN CLUSTERS
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title(f"DBSCAN Clustering ({n_clusters} Clusters, {n_noise} Outliers)")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.show()
