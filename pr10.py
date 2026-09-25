# ==============================================================================
# Unsupervised Learning Technique Practical Manual - Practical No 10
# Aim: To implement Principal Component Analysis (PCA) in Python for visualisation.
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# 1. Iris Dataset PCA 2D Visualization
# ------------------------------------------------------------------------------
print("=================== 1. IRIS PCA VISUALIZATION ===================")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Iris Data (Head):\n", df.head())

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['Species'] = iris.target

# Plot with species names and legend
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for target_id, color, label in zip([0, 1, 2], colors, iris.target_names):
    subset = pca_df[pca_df['Species'] == target_id]
    plt.scatter(subset['PC1'], subset['PC2'], c=color, label=label)

plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA Visualization of Iris Dataset")
plt.legend()
plt.show()


# ------------------------------------------------------------------------------
# PRACTICE QUESTION: Wine Dataset PCA 2D Visualization
# ------------------------------------------------------------------------------
print("\n=================== PRACTICE QUESTION: WINE PCA ===================")
wine = load_wine()
X_wine_scaled = StandardScaler().fit_transform(wine.data)
X_wine_pca = PCA(n_components=2).fit_transform(X_wine_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_wine_pca[:, 0],
    X_wine_pca[:, 1],
    c=wine.target,
    cmap='viridis'
)
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("Wine Dataset - 2D PCA Visualization")
plt.colorbar(label='Class')
plt.show()
