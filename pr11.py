# ==============================================================================
# Unsupervised Learning Technique Practical Manual - Practical No 11
# Aim: IMPLEMENTATION OF GAUSSIAN MIXTURE MODEL (GMM)
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------------------
# 1. Generate Dataset & Visualize Original
# ------------------------------------------------------------------------------
X, y = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.8,
    random_state=42
)

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ------------------------------------------------------------------------------
# 2. Create GMM Model & Fit
# ------------------------------------------------------------------------------
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42
)
gmm.fit(X)


# ------------------------------------------------------------------------------
# 3. Display Hard Labels & Soft Clustering Probabilities
# ------------------------------------------------------------------------------
labels = gmm.predict(X)
print("Cluster Labels (First 10):\n", labels[:10])

# predict_proba returns probability of each point belonging to each cluster
probabilities = gmm.predict_proba(X)
print("\nSoft Probabilities for First Data Point:\n", probabilities[0])


# ------------------------------------------------------------------------------
# 4. Visualize GMM Clusters
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("GMM Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Silhouette Score
score = silhouette_score(X, labels)
print(f"\nOverall Silhouette Score: {score:.4f}")


# ------------------------------------------------------------------------------
# PRACTICE EXERCISES (Iris Dataset: Exercises 1 to 6)
# ------------------------------------------------------------------------------
print("\n=================== PRACTICE EXERCISES ON IRIS ===================")
iris_X, _ = load_iris(return_X_y=True)

# Exercise 1: Apply GMM to Iris with 3 clusters
gmm_ex1 = GaussianMixture(n_components=3, random_state=42)
pred_ex1 = gmm_ex1.fit_predict(iris_X)
print("Exercise 1 (n_components=3): Cluster Counts =", [list(pred_ex1).count(c) for c in range(3)])

# Exercise 2: Change to n_components=2 and compare
gmm_ex2 = GaussianMixture(n_components=2, random_state=42)
pred_ex2 = gmm_ex2.fit_predict(iris_X)
print("Exercise 2 (n_components=2): Cluster Counts =", [list(pred_ex2).count(c) for c in range(2)])

# Exercise 3: Apply GMM with n_components=4
gmm_ex3 = GaussianMixture(n_components=4, random_state=42)
pred_ex3 = gmm_ex3.fit_predict(iris_X)
print("Exercise 3 (n_components=4): Cluster Counts =", [list(pred_ex3).count(c) for c in range(4)])

# Exercise 4: Compare GMM and K-Means on same dataset
gmm_lbl = GaussianMixture(n_components=3, random_state=42).fit_predict(iris_X)
km_lbl = KMeans(n_clusters=3, random_state=42).fit_predict(iris_X)
print(f"\nExercise 4: GMM Score = {silhouette_score(iris_X, gmm_lbl):.4f} | K-Means Score = {silhouette_score(iris_X, km_lbl):.4f}")

# Exercise 5: Silhouette Scores for K = 2, 3, 4, 5 using GMM
print("\nExercise 5: Silhouette Scores for Different K:")
for k in [2, 3, 4, 5]:
    pred_k = GaussianMixture(n_components=k, random_state=42).fit_predict(iris_X)
    print(f"K = {k}: Silhouette Score = {silhouette_score(iris_X, pred_k):.4f}")

# Exercise 6: Experiment with different covariance types
print("\nExercise 6: Covariance Types Comparison:")
for cov in ['full', 'tied', 'diag', 'spherical']:
    model_cov = GaussianMixture(n_components=3, covariance_type=cov, random_state=42)
    pred_cov = model_cov.fit_predict(iris_X)
    print(f"Covariance: {cov:10s} | Silhouette Score: {silhouette_score(iris_X, pred_cov):.4f}")
