# Practical 6: Automated Optimal K Finding on Iris with np.argmax()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(load_iris().data)

k_values = range(2, 11)
scores = []

for k in k_values:
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    print(f"K = {k}, Score = {score:.4f}")

best_k = list(k_values)[np.argmax(scores)]
print(f"\n>>> Optimal K: {best_k} (Highest Score: {max(scores):.4f})")

plt.plot(k_values, scores, marker='o', color='g')
plt.title("Silhouette Score vs K")
plt.xlabel("K"); plt.ylabel("Score")
plt.grid(True)
plt.show()
