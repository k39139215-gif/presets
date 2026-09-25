# ==============================================================================
# Unsupervised Learning Technique Journal - Practical No 5
# Aim: To implement the Elbow Method using Python and Scikit-learn to
#      determine the optimal number of clusters (K) for the K-Means algorithm.
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_wine
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# CODE: 1 (Elbow Method on make_blobs Data)
# ------------------------------------------------------------------------------
print("=================== CODE 1: MAKE_BLOBS ===================")
X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

sse = []
k_values1 = range(1, 11)

for k in k_values1:
    kmeans1 = KMeans(n_clusters=k, random_state=0)
    kmeans1.fit(X1)
    sse.append(kmeans1.inertia_)

plt.plot(k_values1, sse, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method (Code 1)')
plt.show()


# ------------------------------------------------------------------------------
# CODE: 2 (Elbow Method on Wine Dataset)
# ------------------------------------------------------------------------------
print("\n=================== CODE 2: WINE DATASET ===================")
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
print(df_wine.head())

scaler = StandardScaler()
scaled_wine = scaler.fit_transform(df_wine)

wcss2 = []
for k in range(1, 11):
    kmeans2 = KMeans(n_clusters=k, random_state=42)
    kmeans2.fit(scaled_wine)
    wcss2.append(kmeans2.inertia_)

for i, value in enumerate(wcss2, start=1):
    print(f"K = {i}, WCSS = {value:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss2, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method on Wine (Code 2)")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# PRACTICE QUESTION: Customer Dataset (Income & Spending)
# ------------------------------------------------------------------------------
print("\n=================== PRACTICE QUESTION ===================")
data_cust = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df_cust = pd.DataFrame(data_cust)
print("Customer Dataset:\n", df_cust)

# Calculate WCSS for K = 1 to 12 (max points available = 12)
wcss_cust = []
k_range_cust = range(1, len(df_cust) + 1)

for k in k_range_cust:
    km_cust = KMeans(n_clusters=k, random_state=42)
    km_cust.fit(df_cust)
    wcss_cust.append(km_cust.inertia_)
    print(f"K = {k}, WCSS = {km_cust.inertia_:.2f}")

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_range_cust), wcss_cust, marker='o', color='purple')
plt.title("Customer Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# After identifying optimal K = 4, assign cluster labels
optimal_k = 4
df_cust['Cluster'] = KMeans(n_clusters=optimal_k, random_state=42).fit_predict(df_cust[['Income', 'Spending']])
print("\n--- Final Customer Dataset with Assigned Clusters ---\n", df_cust)
