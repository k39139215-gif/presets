import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. DATA LOADING (CSV ya Table)
# Option A: Agar CSV file di ho:
# df = pd.read_csv('filename.csv')
# X = df.select_dtypes(include=np.number).dropna().values

# Option B: Agar numbers/table di ho:
df = pd.DataFrame({
    'Feature1': [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    'Feature2': [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
})
X = df.values

# 2. STANDARDIZE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-MEANS MODEL
k = 3  # Set cluster count
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:\n", labels)
print("\nCluster Centroids:\n", centroids)

# 4. PLOT CLUSTERS
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')
plt.title(f"K-Means Clustering (K={k})")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.show()
