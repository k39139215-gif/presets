import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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

# 3. DENDROGRAM (WARD LINKAGE)
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(7, 4))
dendrogram(linked)
plt.title("Hierarchical Dendrogram (Ward Linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# 4. AGGLOMERATIVE CLUSTERING
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
print("\nHierarchical Cluster Labels:\n", agg_labels)
