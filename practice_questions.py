# ==============================================================================
# ALL PRACTICE QUESTIONS & EXERCISES (ULT LAB MANUALS)
# Contains solutions to all Practice Questions from Practicals 2, 4, 5, 10 & 11
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# PRACTICAL 2 - PRACTICE QUESTION (Student Dataset Operations)
# ==============================================================================
print("=================== PRACTICAL 2: PRACTICE QUESTION ===================")
data_p2 = {
    'Roll_No': [1, 2, 3, 4, 5],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha', 'Karan'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Marks': [85, 72, 90, 68, 88],
    'Attendance': [90, 80, 88, 75, 92]
}
df_p2 = pd.DataFrame(data_p2)

# 1. Display only Name and Marks
print("\n1. Name and Marks only:\n", df_p2[['Name', 'Marks']])

# 2. Students scoring > 75 marks
print("\n2. Students scoring > 75 marks:\n", df_p2[df_p2['Marks'] > 75])

# 3. Sort students by Marks in descending order
print("\n3. Sorted by Marks (Descending):\n", df_p2.sort_values(by='Marks', ascending=False))

# 4. Department-wise: Average Marks, Highest Marks, Student Count
print("\n4. Department-wise Summary:\n", df_p2.groupby('Department').agg(
    Average_Marks=('Marks', 'mean'),
    Highest_Marks=('Marks', 'max'),
    Student_Count=('Roll_No', 'count')
))

# 5. Students with attendance > 85%
print("\n5. Attendance > 85%:\n", df_p2[df_p2['Attendance'] > 85])


# ==============================================================================
# PRACTICAL 4 - PRACTICE QUESTIONS (K-Means on Iris & Wine)
# ==============================================================================
print("\n=================== PRACTICAL 4: PRACTICE QUESTIONS ===================")
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans

# Question 1: Apply K-Means to Iris with K = 3
X_iris, _ = load_iris(return_X_y=True)
km_iris = KMeans(n_clusters=3, random_state=42).fit(X_iris)
print("1. Iris K=3 Cluster Labels:\n", km_iris.labels_)

# Question 2: Apply K-Means to Wine with K = 2, 4, 5 and compare
X_wine = StandardScaler().fit_transform(load_wine().data)
print("\n2. Wine Dataset Cluster Comparison:")
for k in [2, 4, 5]:
    km_w = KMeans(n_clusters=k, random_state=42).fit(X_wine)
    dist = pd.Series(km_w.labels_).value_counts().to_dict()
    print(f"   K = {k} Cluster Sizes: {dist}")


# ==============================================================================
# PRACTICAL 5 - PRACTICE QUESTION (Elbow Method on Customer Data)
# ==============================================================================
print("\n=================== PRACTICAL 5: PRACTICE QUESTION ===================")
data_p5 = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df_p5 = pd.DataFrame(data_p5)

# Calculate WCSS for K = 1 to 12
k_range_p5 = range(1, len(df_p5) + 1)
wcss_p5 = [KMeans(n_clusters=k, random_state=42).fit(df_p5).inertia_ for k in k_range_p5]

# Plot Elbow Curve
plt.figure(figsize=(7, 4))
plt.plot(list(k_range_p5), wcss_p5, marker='o', color='purple')
plt.title("Practical 5: Customer Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Assign cluster label using Optimal K = 4
df_p5['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(df_p5[['Income', 'Spending']])
print("Clustered Customer Dataset (Optimal K=4):\n", df_p5)


# ==============================================================================
# PRACTICAL 10 - PRACTICE QUESTION (Wine Dataset 2D PCA Visualization)
# ==============================================================================
print("\n=================== PRACTICAL 10: PRACTICE QUESTION ===================")
from sklearn.decomposition import PCA

wine = load_wine()
X_wine_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(wine.data))

plt.figure(figsize=(7, 4))
plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=wine.target, cmap='viridis')
plt.title("Practical 10: Wine 2D PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Class')
plt.show()


# ==============================================================================
# PRACTICAL 11 - PRACTICE EXERCISES (GMM on Iris)
# ==============================================================================
print("\n=================== PRACTICAL 11: PRACTICE EXERCISES ===================")
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Exercises 1, 2, 3: n_components = 3, 2, 4
for n in [3, 2, 4]:
    pred_gmm = GaussianMixture(n_components=n, random_state=42).fit_predict(X_iris)
    print(f"GMM n_components={n}: Cluster Counts = {[list(pred_gmm).count(c) for c in range(n)]}")

# Exercise 4: Compare GMM vs K-Means
gmm_sc = silhouette_score(X_iris, GaussianMixture(n_components=3, random_state=42).fit_predict(X_iris))
km_sc = silhouette_score(X_iris, KMeans(n_clusters=3, random_state=42).fit_predict(X_iris))
print(f"\nExercise 4 Comparison: GMM Score = {gmm_sc:.4f} | K-Means Score = {km_sc:.4f}")

# Exercise 5: Silhouette Scores for K = 2, 3, 4, 5 using GMM
print("\nExercise 5 Silhouette Scores:")
for k in [2, 3, 4, 5]:
    sc = silhouette_score(X_iris, GaussianMixture(n_components=k, random_state=42).fit_predict(X_iris))
    print(f"   K = {k}: Silhouette Score = {sc:.4f}")

# Exercise 6: Covariance Types Comparison
print("\nExercise 6 Covariance Types:")
for cov in ['full', 'tied', 'diag', 'spherical']:
    sc = silhouette_score(X_iris, GaussianMixture(n_components=3, covariance_type=cov, random_state=42).fit_predict(X_iris))
    print(f"   Covariance: {cov:10s} | Silhouette Score = {sc:.4f}")
