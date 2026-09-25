# Practical 4: K-Means on Wine Dataset with StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
scaled_data = StandardScaler().fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_data)
df['Cluster'] = kmeans.labels_
print("Head with Cluster Labels:\n", df.head())

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Wine K-Means (K=3)")
plt.show()
