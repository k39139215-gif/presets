# ==============================================================================
# Unsupervised Learning Technique Journal - Practical No 4
# Aim: To implement the K-Means clustering algorithm using Python and
#      Scikit-learn to group similar data points into clusters.
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# CODE: 1 (K-Means on 2D Toy Data)
# ------------------------------------------------------------------------------
print("=================== CODE 1: TOY DATA ===================")
# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
k = 2  # Example: 2 clusters
kmeans = KMeans(n_clusters=k, random_state=0)  # Initialize KMeans
kmeans.fit(X)                                  # Train the model on the data

# Get the cluster centers
centroids = kmeans.cluster_centers_
print("Cluster Centers:\n", centroids)

# Predict cluster labels for each data point
labels = kmeans.predict(X)
print("Cluster Labels:\n", labels)

# Visualisation of clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, linewidths=3, color='r', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans clustering (Code 1)')
plt.show()


# ------------------------------------------------------------------------------
# CODE: 2 (K-Means on Wine Dataset)
# ------------------------------------------------------------------------------
print("\n=================== CODE 2: WINE DATASET ===================")
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
print("Wine Head:\n", df.head())
print("Wine Shape:", df.shape)
print("\nWine Describe:\n", df.describe())

# Standardise Dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("\nScaled Data (First 5 Rows):\n", scaled_data[:5])

# Apply K-means clustering
kmeans_wine = KMeans(n_clusters=3, random_state=42)
kmeans_wine.fit(scaled_data)

labels_wine = kmeans_wine.labels_
print("\nWine Cluster Labels:\n", labels_wine)

# Add cluster labels to DataFrame
df['Cluster'] = labels_wine
print("\nDataFrame with Cluster Labels:\n", df.head())

# Display cluster centers
print("\nWine Cluster Centers:\n", kmeans_wine.cluster_centers_)

# Visualise the cluster (Alcohol vs Malic Acid)
plt.figure(figsize=(8, 6))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels_wine, cmap='viridis')
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("K-Means Clustering on Wine (Code 2)")
plt.show()


# ------------------------------------------------------------------------------
# PRACTICE QUESTION 1: Iris Dataset with K = 3
# ------------------------------------------------------------------------------
print("\n=================== PRACTICE QUESTION 1 ===================")
X_iris, _ = load_iris(return_X_y=True)
kmeans_iris = KMeans(n_clusters=3, random_state=42).fit(X_iris)
print("Iris Cluster Labels (K=3):\n", kmeans_iris.labels_)
print("Iris Cluster Centers:\n", kmeans_iris.cluster_centers_)


# ------------------------------------------------------------------------------
# PRACTICE QUESTION 2: Wine Dataset with K = 2, 4, and 5 Comparison
# ------------------------------------------------------------------------------
print("\n=================== PRACTICE QUESTION 2 ===================")
for k_val in [2, 4, 5]:
    km_comp = KMeans(n_clusters=k_val, random_state=42).fit(scaled_data)
    dist = pd.Series(km_comp.labels_).value_counts().to_dict()
    print(f"K = {k_val} Cluster Assignment Distribution: {dist}")
