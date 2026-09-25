# ==============================================================================
# PRACTICAL 6: SILHOUETTE SCORE METHOD FOR OPTIMAL K
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# 1. Load and Scale Dataset
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

# 2. Test K from 2 to 10
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

# 3. Automatically Identify Optimal K (Highest Score)
best_idx = np.argmax(silhouette_scores)
best_k = list(k_values)[best_idx]
print(f"\n>>> Optimal K = {best_k} with Highest Score = {silhouette_scores[best_idx]:.4f}")

# 4. Plot Silhouette Curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker='o', color='g')
plt.title("Silhouette Score vs Number of Clusters (K)")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
