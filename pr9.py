# ==============================================================================
# PRACTICAL 9: PRINCIPAL COMPONENT ANALYSIS (PCA) FOR DIMENSION REDUCTION
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Dataset (13 dimensions)
wine = load_wine()
print("Original Dimensions:", wine.data.shape)

# 2. Standardize Features
X_scaled = StandardScaler().fit_transform(wine.data)

# 3. Reduce from 13 Features to 2 Principal Components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Reduced Dimensions:", X_pca.shape)

# 4. Variance Explanation
print("Explained Variance Ratio (PC1, PC2):", pca.explained_variance_ratio_)
total_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total Information Retained: {total_var:.2f}%")

# 5. Scatter Plot of 2D Projections
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine.target, cmap='viridis')
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA Dimensionality Reduction (13D -> 2D)")
plt.colorbar(label='Wine Class')
plt.show()
