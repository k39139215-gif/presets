# Practical 5: Elbow Method on Wine Dataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)

wcss = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    wcss.append(km.inertia_)
    print(f"K = {k}, WCSS = {km.inertia_:.2f}")

plt.plot(k_values, wcss, marker='o')
plt.title("Wine Dataset Elbow Method")
plt.xlabel("K"); plt.ylabel("WCSS")
plt.grid(True)
plt.show()
