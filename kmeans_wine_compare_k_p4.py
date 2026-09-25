# Practical 4: Practice Question 2 (Wine Dataset K=2, 4, 5 Comparison)
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)

for k in [2, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    print(f"K={k} Cluster Distribution:", pd.Series(km.labels_).value_counts().to_dict())
