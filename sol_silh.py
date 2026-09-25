import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. DATA LOADING
# df = pd.read_csv('filename.csv')
# X = df.select_dtypes(include=np.number).dropna().values
df = pd.DataFrame({
    'Feature1': [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    'Feature2': [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
})
X = df.values

# 2. STANDARDIZE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. SILHOUETTE SCORES
scores = []
k_range = range(2, min(7, len(X_scaled)))
print("--- Silhouette Scores ---")
for i in k_range:
    km = KMeans(n_clusters=i, random_state=42)
    labels = km.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, labels)
    scores.append(s)
    print(f"K = {i}: Silhouette Score = {s:.4f}")

best_k = list(k_range)[np.argmax(scores)]
print(f"\nOptimal K identified: {best_k}")

# 4. PLOT SILHOUETTE SCORES
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), scores, marker='o', color='green')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
