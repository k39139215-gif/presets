# Practical 11: Gaussian Mixture Model (GMM) - Soft Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(X)

print("Hard Labels (First 5):", gmm.predict(X)[:5])
print("\nSoft Probabilities (Sample 0):\n", gmm.predict_proba(X)[0])

plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap='viridis')
plt.title("GMM Clustering (3 Components)")
plt.show()
