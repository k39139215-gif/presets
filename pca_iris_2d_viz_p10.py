# Practical 10: PCA 2D Visualization with Legends (Iris)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(iris.data))

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Species'] = iris.target

colors = ['r', 'g', 'b']
for target_id, col, name in zip([0, 1, 2], colors, iris.target_names):
    sub = pca_df[pca_df['Species'] == target_id]
    plt.scatter(sub['PC1'], sub['PC2'], c=col, label=name)

plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Iris 2D PCA Projection")
plt.legend()
plt.show()
