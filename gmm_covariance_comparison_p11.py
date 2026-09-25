# Practical 11: GMM Covariance Types Comparison (full, tied, diag, spherical)
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X, _ = load_iris(return_X_y=True)

print("--- GMM Covariance Types Comparison ---")
for cov in ['full', 'tied', 'diag', 'spherical']:
    model = GaussianMixture(n_components=3, covariance_type=cov, random_state=42)
    labels = model.fit_predict(X)
    print(f"Covariance: {cov:10s} | Silhouette Score: {silhouette_score(X, labels):.4f}")
