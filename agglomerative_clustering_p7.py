# Practical 7: Agglomerative Clustering (K=3) & Scatter Plot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)
X_scaled = StandardScaler().fit_transform(df)

model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df['Cluster'] = model.fit_predict(X_scaled)
print(df)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis', s=80)
plt.title("Agglomerative Clustering (K=3)")
plt.xlabel("Income (Scaled)")
plt.ylabel("Spending (Scaled)")
plt.show()
