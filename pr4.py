# ==============================================================================
# PRACTICAL 4: K-MEANS CLUSTERING
# ==============================================================================
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PART 1: K-Means on Synthetic 2D Points ---
import numpy as np
X_toy = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
km_toy = KMeans(n_clusters=2, random_state=0).fit(X_toy)

plt.scatter(X_toy[:, 0], X_toy[:, 1], c=km_toy.labels_, cmap='viridis')
plt.scatter(km_toy.cluster_centers_[:, 0], km_toy.cluster_centers_[:, 1], s=200, color='r', marker='X')
plt.title("Toy K-Means (K=2)")
plt.show()

# --- PART 2: Practice Question 1 (Iris Dataset with K=3) ---
X_iris, _ = load_iris(return_X_y=True)
km_iris = KMeans(n_clusters=3, random_state=42).fit(X_iris)
print("Iris (K=3) Cluster Labels:\n", km_iris.labels_)

# --- PART 3: Practice Question 2 (Wine Dataset with K=2, 4, 5 Comparison) ---
wine = load_wine()
X_wine = StandardScaler().fit_transform(wine.data)

for k in [2, 4, 5]:
    km_wine = KMeans(n_clusters=k, random_state=42).fit(X_wine)
    print(f"Wine K={k} Cluster Distribution:", pd.Series(km_wine.labels_).value_counts().to_dict())
