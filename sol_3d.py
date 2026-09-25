import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. LOAD DATA (Built-in Iris ya Custom Table)
iris = load_iris()
X = iris.data
y = iris.target

# 2. STANDARDIZE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA REDUCTION TO 3 DIMENSIONS
pca = PCA(n_components=3)  # 3D ke liye 3 components
X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", X_scaled.shape)
print("3D Reduced Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

# 4. 3D SCATTER PLOT
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')  # 3D projection enable kiya

scatter = ax.scatter(
    X_pca[:, 0],  # X axis: PC1
    X_pca[:, 1],  # Y axis: PC2
    X_pca[:, 2],  # Z axis: PC3
    c=y,
    cmap='viridis',
    s=50
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA Visualization")
plt.show()
