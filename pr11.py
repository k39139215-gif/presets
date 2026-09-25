# ==============================================================================
# PRACTICAL 11: GAUSSIAN MIXTURE MODEL (GMM) - SOFT CLUSTERING
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# --- PART 1: GMM on Synthetic Data with Soft Probabilities ---
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

labels = gmm.predict(X)              # Hard assignments
probs = gmm.predict_proba(X)         # Soft clustering probabilities

print("Sample 0 Hard Cluster:", labels[0])
print("Sample 0 Probabilities (Soft Clustering):\n", probs[0])

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)
plt.title("GMM Clustering (3 Components)")
plt.show()

# --- PART 2: Practice Exercises on Iris Dataset ---
iris_X, _ = load_iris(return_X_y=True)

# Exercise 5: Silhouette Scores for K = 2, 3, 4, 5
print("\n--- GMM Silhouette Scores ---")
for k in [2, 3, 4, 5]:
    pred = GaussianMixture(n_components=k, random_state=42).fit_predict(iris_X)
    print(f"K = {k}: Silhouette Score = {silhouette_score(iris_X, pred):.4f}")

# Exercise 6: Covariance Types Comparison
print("\n--- Covariance Types Comparison (K=3) ---")
for cov in ['full', 'tied', 'diag', 'spherical']:
    pred = GaussianMixture(n_components=3, covariance_type=cov, random_state=42).fit_predict(iris_X)
    print(f"Covariance: {cov:10s} | Silhouette Score = {silhouette_score(iris_X, pred):.4f}")
