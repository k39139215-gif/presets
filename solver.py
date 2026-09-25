# ==============================================================================
# MASTER SOLVER TEMPLATES (STANDARD EXAM SYNTAX - NO CUSTOM WRAPPERS)
# 100% Standard Scikit-Learn code jo screen par run bhi hoga aur 
# exact yahi code aap apni answer sheet / paper par bhi likh sakte ho!
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# STEP 1: DATA LOADING (Teacher ka dataset yahan aayega)
# ------------------------------------------------------------------------------
# OPTION A: Agar teacher ne CSV file upload karne ko bola ho:
# df = pd.read_csv('filename.csv')
# X = df.select_dtypes(include=np.number).dropna().values

# OPTION B: Agar teacher ne table ya custom numbers diye hon:
df = pd.DataFrame({
    'Feature1': [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    'Feature2': [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
})
X = df.values

# Standardize data (Sabhi algorithms ke liye mandatory)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data Scaled Successfully. Shape:", X_scaled.shape)


# ------------------------------------------------------------------------------
# TASK 1: K-MEANS CLUSTERING
# ------------------------------------------------------------------------------
from sklearn.cluster import KMeans

k = 3  # Clusters count
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("\n--- K-Means Output ---")
print("Cluster Labels:\n", labels)
print("Cluster Centroids:\n", centroids)

plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')
plt.title(f"K-Means Clustering (K={k})")
plt.show()


# ------------------------------------------------------------------------------
# TASK 2: ELBOW METHOD FOR OPTIMAL K
# ------------------------------------------------------------------------------
wcss = []
k_range = range(1, min(11, len(X_scaled) + 1))
for i in k_range:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(k_range), wcss, marker='o', color='blue')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# TASK 3: SILHOUETTE SCORE METHOD
# ------------------------------------------------------------------------------
from sklearn.metrics import silhouette_score

silhouette_scores = []
k_values = range(2, min(7, len(X_scaled)))
print("\n--- Silhouette Scores ---")
for i in k_values:
    km = KMeans(n_clusters=i, random_state=42)
    cl_labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cl_labels)
    silhouette_scores.append(score)
    print(f"K = {i}: Silhouette Score = {score:.4f}")

best_k = list(k_values)[np.argmax(silhouette_scores)]
print(f"\nOptimal K identified: {best_k}")

plt.figure(figsize=(6, 4))
plt.plot(list(k_values), silhouette_scores, marker='o', color='green')
plt.title("Silhouette Score vs K")
plt.xlabel("K"); plt.ylabel("Score")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# TASK 4: HIERARCHICAL CLUSTERING & DENDROGRAM
# ------------------------------------------------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Linkage Matrix (Ward Linkage)
linked = linkage(X_scaled, method='ward')

# Plot Dendrogram
plt.figure(figsize=(7, 4))
dendrogram(linked)
plt.title("Hierarchical Dendrogram (Ward Linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
print("\nHierarchical Cluster Labels:\n", agg_labels)


# ------------------------------------------------------------------------------
# TASK 5: DBSCAN CLUSTERING & NOISE DETECTION
# ------------------------------------------------------------------------------
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=3)
db_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(db_labels) - {-1})
n_noise = list(db_labels).count(-1)

print("\n--- DBSCAN Output ---")
print(f"Total Clusters: {n_clusters}")
print(f"Noise / Outliers (-1): {n_noise}")

plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=db_labels, cmap='viridis')
plt.title(f"DBSCAN Clustering ({n_clusters} Clusters, {n_noise} Outliers)")
plt.show()


# ------------------------------------------------------------------------------
# TASK 6: PCA DIMENSION REDUCTION & 2D PROJECTION
# ------------------------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n--- PCA Output ---")
print("Original Shape:", X_scaled.shape)
print("Reduced Shape (2D):", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA 2D Projection")
plt.show()


# ------------------------------------------------------------------------------
# TASK 7: GAUSSIAN MIXTURE MODEL (GMM) - SOFT CLUSTERING
# ------------------------------------------------------------------------------
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

hard_labels = gmm.predict(X_scaled)
soft_probabilities = gmm.predict_proba(X_scaled)

print("\n--- GMM Output ---")
print("Hard Cluster Labels (First 5):\n", hard_labels[:5])
print("Soft Probabilities (First Sample):\n", soft_probabilities[0])

plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hard_labels, cmap='viridis')
plt.title("GMM Clustering (Soft Assignment)")
plt.show()
