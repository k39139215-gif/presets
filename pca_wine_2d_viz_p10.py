# Practical 10: Practice Question (PCA 2D Visualization on Wine)
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(wine.data))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine.target, cmap='viridis')
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Wine 2D PCA Visualization")
plt.colorbar(label='Class')
plt.show()
