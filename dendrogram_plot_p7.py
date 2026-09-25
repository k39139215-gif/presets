# Practical 7: Hierarchical Clustering Dendrogram (Ward Linkage)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)
X_scaled = StandardScaler().fit_transform(df)

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(9, 5))
dendrogram(linked, labels=df.index + 1)
plt.title("Hierarchical Dendrogram (Ward)")
plt.xlabel("Customer ID")
plt.ylabel("Distance")
plt.show()
