# Practical 9: PCA for Dimension Reduction (Wine 13D to 2D)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
X_scaled = StandardScaler().fit_transform(wine.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", wine.data.shape)
print("Reduced Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Information Retained:", round(np.sum(pca.explained_variance_ratio_) * 100, 2), "%")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine.target, cmap='viridis')
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA Dimension Reduction on Wine")
plt.colorbar(label='Wine Class')
plt.show()
