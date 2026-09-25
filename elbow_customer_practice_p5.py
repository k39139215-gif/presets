# Practical 5: Practice Question (Customer Income & Spending Dataset)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)

k_values = range(1, len(df) + 1)
wcss = [KMeans(n_clusters=k, random_state=42).fit(df).inertia_ for k in k_values]

plt.plot(list(k_values), wcss, marker='o')
plt.title("Elbow Method (Optimal K=4)")
plt.xlabel("K"); plt.ylabel("WCSS")
plt.grid(True)
plt.show()

df['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(df[['Income', 'Spending']])
print("\n--- Clustered Customer Data ---\n", df)
