# ==============================================================================
# PRACTICAL 10: PCA FOR 2D VISUALIZATION
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- PART 1: Iris Dataset PCA 2D Plot with Legends ---
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Species'] = iris.target

plt.figure(figsize=(7, 5))
colors = ['red', 'green', 'blue']
for target_id, color, name in zip([0, 1, 2], colors, iris.target_names):
    sub = pca_df[pca_df['Species'] == target_id]
    plt.scatter(sub['PC1'], sub['PC2'], c=color, label=name)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris Dataset - 2D PCA Visualization")
plt.legend()
plt.show()

# --- PART 2: Practice Question (Wine Dataset PCA 2D Plot) ---
wine = load_wine()
X_wine_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(wine.data))

plt.figure(figsize=(7, 5))
plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=wine.target, cmap='viridis')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Dataset - 2D PCA Visualization")
plt.colorbar(label='Class')
plt.show()
