# Practical 5: Elbow Method on Synthetic Data (make_blobs)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

sse = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    sse.append(km.inertia_)

plt.plot(k_values, sse, 'bo-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE / WCSS")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()
