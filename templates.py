# ==============================================================================
# UNIVERSAL PLUG-AND-PLAY TEMPLATES FOR ANY CUSTOM DATASET
# Teacher chahe koi bhi naya dataset ya table de, bas Step 0 me data daalo!
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# STEP 0: APNA DATA LOAD KARO (Choose Option A or Option B)
# ------------------------------------------------------------------------------
# OPTION A: Agar teacher ne CSV file upload karne ko bola ho:
# df = pd.read_csv('teacher_file.csv')
# X = df.select_dtypes(include=np.number).dropna() # Sirf numeric values

# OPTION B: Agar teacher ne khud board par table/numbers diye hon:
df = pd.DataFrame({
    "Feature1": [10, 15, 20, 50, 55, 60, 85, 90],
    "Feature2": [25, 30, 35, 70, 75, 80, 15, 20]
})
X = df.values

# Standardize data (Sabhi algorithms ke liye recommended):
X_scaled = StandardScaler().fit_transform(X)
print("Data Ready. Shape:", X_scaled.shape)


# ------------------------------------------------------------------------------
# TEMPLATE 1: K-MEANS CLUSTERING (ON ANY DATA)
# ------------------------------------------------------------------------------
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
print("\n[K-Means] Cluster Labels:\n", km.labels_)
print("[K-Means] Centroids:\n", km.cluster_centers_)


# ------------------------------------------------------------------------------
# TEMPLATE 2: ELBOW METHOD FOR OPTIMAL K (ON ANY DATA)
# ------------------------------------------------------------------------------
k_range = range(1, min(11, len(X_scaled) + 1))
wcss = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_ for k in k_range]

plt.figure(figsize=(6, 4))
plt.plot(list(k_range), wcss, marker='o', color='blue')
plt.title("Elbow Curve")
plt.xlabel("K"); plt.ylabel("WCSS")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# TEMPLATE 3: SILHOUETTE SCORE METHOD (ON ANY DATA)
# ------------------------------------------------------------------------------
from sklearn.metrics import silhouette_score

print("\n[Silhouette Scores]:")
for k in range(2, min(6, len(X_scaled))):
    lbl = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
    print(f"K = {k}: Silhouette Score = {silhouette_score(X_scaled, lbl):.4f}")


# ------------------------------------------------------------------------------
# TEMPLATE 4: HIERARCHICAL CLUSTERING & DENDROGRAM (ON ANY DATA)
# ------------------------------------------------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Plot Dendrogram
plt.figure(figsize=(7, 4))
dendrogram(linkage(X_scaled, method='ward'))
plt.title("Dendrogram (Ward Linkage)")
plt.show()

# Fit Agglomerative Model
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
print("\n[Hierarchical] Cluster Labels:\n", agg_labels)


# ------------------------------------------------------------------------------
# TEMPLATE 5: DBSCAN CLUSTERING & NOISE DETECTION (ON ANY DATA)
# ------------------------------------------------------------------------------
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=2).fit(X_scaled)
n_clusters = len(set(db.labels_) - {-1})
n_noise = list(db.labels_).count(-1)

print(f"\n[DBSCAN] Total Clusters: {n_clusters}")
print(f"[DBSCAN] Noise Points (-1): {n_noise}")
print("[DBSCAN] Labels:", db.labels_)


# ------------------------------------------------------------------------------
# TEMPLATE 6: PCA DIMENSION REDUCTION (ON ANY DATA)
# ------------------------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\n[PCA] Reduced Shape (2D):", X_pca.shape)
print("[PCA] Explained Variance Ratio:", pca.explained_variance_ratio_)
print(f"[PCA] Total Information Retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap='viridis')
plt.title("PCA 2D Projection")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()


# ------------------------------------------------------------------------------
# TEMPLATE 7: GAUSSIAN MIXTURE MODEL (SOFT CLUSTERING) (ON ANY DATA)
# ------------------------------------------------------------------------------
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42).fit(X_scaled)
print("\n[GMM] Hard Cluster Labels:\n", gmm.predict(X_scaled))
print("\n[GMM] Soft Probabilities (First 3 samples):\n", gmm.predict_proba(X_scaled)[:3])
