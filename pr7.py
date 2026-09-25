# ==============================================================================
# Unsupervised Learning Technique Practical Manual - Practical No 7
# Aim: To implement Hierarchical Clustering using Python and Scikit-learn
#      and visualize the resulting clusters using a dendrogram.
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------------------------
# 1. Create the Dataset
# ------------------------------------------------------------------------------
data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)


# ------------------------------------------------------------------------------
# 2. Standardise the Data
# ------------------------------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# ------------------------------------------------------------------------------
# 3. Generate the Linkage Matrix (Ward Linkage)
# ------------------------------------------------------------------------------
linked = linkage(scaled_data, method='ward')


# ------------------------------------------------------------------------------
# 4. Plot the Dendrogram
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
dendrogram(
    linked,
    labels=df.index + 1
)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Distance")
plt.show()


# ------------------------------------------------------------------------------
# 5. Apply Agglomerative Clustering (K = 3)
# ------------------------------------------------------------------------------
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)
labels = model.fit_predict(scaled_data)


# ------------------------------------------------------------------------------
# 6. Add Cluster Labels to the Dataset
# ------------------------------------------------------------------------------
df["Cluster"] = labels
print("\nClustered Dataset:\n", df)


# ------------------------------------------------------------------------------
# 7. Visualize the Clusters
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=labels,
    cmap='viridis',
    s=100
)
plt.xlabel("Income (Standardized)")
plt.ylabel("Spending Score (Standardized)")
plt.title("Hierarchical Clustering")
plt.show()
