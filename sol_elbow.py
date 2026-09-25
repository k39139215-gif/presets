import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

# 3. ELBOW METHOD (WCSS / INERTIA)
wcss = []
k_range = range(1, min(11, len(X_scaled) + 1))
for i in k_range:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# 4. PLOT ELBOW CURVE
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), wcss, marker='o', color='blue')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.show()
