# ==============================================================================
# Unsupervised Learning Technique Journal - Practical No 6
# Aim: To implement the Silhouette Score Method using Python and Scikit-learn
#      to determine the optimal number of clusters (K) for K-Means.
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# ------------------------------------------------------------------------------
# 1. Basic Example: 3 Clusters Toy Array
# ------------------------------------------------------------------------------
print("=================== 1. TOY DATA SILHOUETTE ===================")
X_toy = np.array([
    [2, 5], [3, 4], [4, 6],    # Cluster 1
    [8, 3], [9, 2], [10, 5],   # Cluster 2
    [6, 10], [7, 8], [8, 9]    # Cluster 3
])

kmeans_toy = KMeans(n_clusters=3, random_state=0)
kmeans_toy.fit(X_toy)
score_toy = silhouette_score(X_toy, kmeans_toy.labels_)
print(f"Silhouette score: {score_toy:.3f}")


# ------------------------------------------------------------------------------
# 2. CODE 1: Synthetic Data with Overall & Per-Point Silhouette Values
# ------------------------------------------------------------------------------
print("\n=================== CODE 1: PER-POINT SCORES ===================")
X_blobs, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

kmeans_blobs = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_blobs = kmeans_blobs.fit_predict(X_blobs)

score_overall = silhouette_score(X_blobs, labels_blobs)
print(f"Overall silhouette score: {score_overall:.3f}")

sample_scores = silhouette_samples(X_blobs, labels_blobs)
print(f"First 5 point scores: {sample_scores[:5]}")


# ------------------------------------------------------------------------------
# 3. CODE 2: Testing K from 2 to 10 on Synthetic Blobs
# ------------------------------------------------------------------------------
print("\n=================== CODE 2: K=2 TO 10 ON BLOBS ===================")
X_b300, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
silhouette_scores_b = []
k_range_b = range(2, 11)

for k in k_range_b:
    km_b = KMeans(n_clusters=k, random_state=0, n_init=10)
    cl_labels = km_b.fit_predict(X_b300)
    sc = silhouette_score(X_b300, cl_labels)
    silhouette_scores_b.append(sc)
    print(f"For k={k}, Silhouette Score: {sc:.3f}")

plt.plot(k_range_b, silhouette_scores_b)
plt.title("Silhouette Scores for different K values (Blobs)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(k_range_b)
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# 4. CODE 3: Iris Dataset & Automated Optimal K with np.argmax()
# ------------------------------------------------------------------------------
print("\n=================== CODE 3: IRIS OPTIMAL K ===================")
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

scaler = StandardScaler()
scaled_iris = scaler.fit_transform(df_iris)

silhouette_scores_iris = []
k_values_iris = range(2, 11)

for k in k_values_iris:
    km_iris = KMeans(n_clusters=k, random_state=42)
    lbls = km_iris.fit_predict(scaled_iris)
    sc = silhouette_score(scaled_iris, lbls)
    silhouette_scores_iris.append(sc)

for k, sc in zip(k_values_iris, silhouette_scores_iris):
    print(f"K = {k}, Silhouette Score = {sc:.4f}")

best_index = np.argmax(silhouette_scores_iris)
best_k = list(k_values_iris)[best_index]
best_score = silhouette_scores_iris[best_index]

print(f"\nOptimal K: {best_k}")
print(f"Highest Silhouette Score: {best_score:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(k_values_iris, silhouette_scores_iris, marker='o', color='green')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal K (Iris)")
plt.grid(True)
plt.show()
