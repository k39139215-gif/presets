# ==============================================================================
# PRACTICAL 7: HIERARCHICAL CLUSTERING & DENDROGRAM
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Dataset
data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)

# 2. Standardize
X_scaled = StandardScaler().fit_transform(df)

# 3. Create Linkage Matrix
linked = linkage(X_scaled, method='ward')

# 4. Plot Dendrogram
plt.figure(figsize=(9, 5))
dendrogram(linked, labels=df.index + 1)
plt.title("Hierarchical Dendrogram (Ward Linkage)")
plt.xlabel("Customer ID")
plt.ylabel("Distance")
plt.show()

# 5. Fit Agglomerative Model with K=3
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df['Cluster'] = model.fit_predict(X_scaled)
print("--- Clustered Dataset ---\n", df)

# 6. Scatter Plot
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis', s=80)
plt.title("Hierarchical Clustering (K=3)")
plt.xlabel("Income (Standardized)")
plt.ylabel("Spending (Standardized)")
plt.show()
