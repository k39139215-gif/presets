# ==============================================================================
# PRACTICAL 5: ELBOW METHOD FOR OPTIMAL K
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Practice Question Dataset
data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)

# 2. Calculate WCSS for K = 1 to 12 (dataset has 12 samples)
k_values = range(1, len(df) + 1)
wcss = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df)
    wcss.append(km.inertia_)
    print(f"K = {k}, WCSS = {km.inertia_:.2f}")

# 3. Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), wcss, marker='o', color='b')
plt.title("Elbow Method for Customer Dataset")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.show()

# 4. Apply K-Means with Optimal K (Elbow at K=4) & Assign Labels
optimal_k = 4
df['Cluster'] = KMeans(n_clusters=optimal_k, random_state=42).fit_predict(df[['Income', 'Spending']])
print("\n--- Final Clustered Customer Data ---\n", df)
