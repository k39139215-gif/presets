import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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

# 3. GAUSSIAN MIXTURE MODEL (GMM)
k = 3
gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

hard_labels = gmm.predict(X_scaled)
soft_probs = gmm.predict_proba(X_scaled)

print("Hard Cluster Labels:\n", hard_labels)
print("\nSoft Probabilities (Sample 0):\n", np.round(soft_probs[0], 4))

# 4. PLOT GMM CLUSTERS
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hard_labels, cmap='viridis')
plt.title(f"GMM Clustering (K={k})")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.show()
