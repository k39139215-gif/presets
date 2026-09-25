import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. DATA LOADING
# df = pd.read_csv('filename.csv')
# X = df.select_dtypes(include=np.number).dropna().values
df = pd.DataFrame({
    'Feature1': [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    'Feature2': [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
})
X = df.values

# 2. STANDARDIZE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA DIMENSION REDUCTION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Data Shape:", X_scaled.shape)
print("PCA Reduced Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

# 4. PLOT PCA 2D PROJECTION
plt.figure(figsize=(6, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='purple')
plt.title("PCA 2D Projection")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(True)
plt.show()
