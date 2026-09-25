# ==============================================================================
# Unsupervised Learning Technique Practical Manual - Practical No 9
# Aim: To implement Principal Component Analysis (PCA) in Python for
#      reducing the dimensions of a dataset.
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# 1. Load Dataset & Check Dimensions
# ------------------------------------------------------------------------------
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
print("Original shape:", df.shape)


# ------------------------------------------------------------------------------
# 2. Standardise the Data
# ------------------------------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# ------------------------------------------------------------------------------
# 3. Apply PCA (Reduce 13 features to 2 Principal Components)
# ------------------------------------------------------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print("Reduced shape:", pca_data.shape)


# ------------------------------------------------------------------------------
# 4. View the PCA Data
# ------------------------------------------------------------------------------
print("\nPCA Data (First 5 Rows):\n", pca_data[:5])


# ------------------------------------------------------------------------------
# 5. Check Explained Variance Ratio
# ------------------------------------------------------------------------------
print("\nExplained Variance Ratio:")
print(f"PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
total_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total Variance Retained: {total_var:.2f}%")


# ------------------------------------------------------------------------------
# 6. Visualize Using Wine Classes
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=wine.target,
    cmap='viridis'
)
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA on Wine Dataset (13D -> 2D)")
plt.colorbar(label='Wine Class')
plt.show()
