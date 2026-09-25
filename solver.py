# ==============================================================================
# KARTIK'S UNIVERSAL ULT AUTO-SOLVER & SWISS ARMY KNIFE
# Ye ek master toolkit hai jo kisi bhi dataset par 1 line me koi bhi algorithm
# chala deta hai, chahe teacher kuch bhi twist kare!
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------------------------
# 1. AUTO-DATA PREPARATION (Handles CSV, DataFrames, Arrays & Missing Values)
# ------------------------------------------------------------------------------
def prepare(data):
    """Kisi bhi type ke data ko clean, numeric aur standardized bana deta hai"""
    if isinstance(data, str):
        data = pd.read_csv(data)
    if isinstance(data, pd.DataFrame):
        data = data.select_dtypes(include=np.number)
        data = data.fillna(data.mean())
        data = data.values
    return StandardScaler().fit_transform(data)


# ------------------------------------------------------------------------------
# 2. UNIVERSAL K-MEANS SOLVER
# ------------------------------------------------------------------------------
def do_kmeans(data, k=3):
    X = prepare(data)
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    print(f"=== K-Means Clustering (K={k}) ===")
    print("Labels:\n", km.labels_)
    print("\nCentroids:\n", km.cluster_centers_)
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, c='red', marker='X')
    plt.title(f"K-Means Clustering (K={k})")
    plt.show()
    return km


# ------------------------------------------------------------------------------
# 3. UNIVERSAL ELBOW METHOD SOLVER
# ------------------------------------------------------------------------------
def do_elbow(data, max_k=10):
    X = prepare(data)
    max_k = min(max_k, len(X))
    wcss = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in range(1, max_k + 1)]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o', color='b')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------------------------
# 4. UNIVERSAL SILHOUETTE SCORE SOLVER
# ------------------------------------------------------------------------------
def do_silhouette(data, max_k=6):
    X = prepare(data)
    scores = []
    print("=== Silhouette Scores for Different K ===")
    for k in range(2, min(max_k + 1, len(X))):
        lbl = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        sc = silhouette_score(X, lbl)
        scores.append((k, sc))
        print(f"K = {k}: Silhouette Score = {sc:.4f}")
    best_k, best_score = max(scores, key=lambda item: item[1])
    print(f"\n>>> Optimal K: {best_k} (Highest Score: {best_score:.4f})")


# ------------------------------------------------------------------------------
# 5. UNIVERSAL HIERARCHICAL & DENDROGRAM SOLVER
# ------------------------------------------------------------------------------
def do_hierarchical(data, k=3, method='ward'):
    X = prepare(data)
    plt.figure(figsize=(7, 4))
    dendrogram(linkage(X, method=method))
    plt.title(f"Dendrogram ({method.capitalize()} Linkage)")
    plt.show()
    
    agg = AgglomerativeClustering(n_clusters=k, linkage=method)
    labels = agg.fit_predict(X)
    print(f"=== Hierarchical Clustering (K={k}) ===")
    print("Cluster Labels:\n", labels)
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(f"Hierarchical Clusters (K={k})")
    plt.show()
    return labels


# ------------------------------------------------------------------------------
# 6. UNIVERSAL DBSCAN SOLVER
# ------------------------------------------------------------------------------
def do_dbscan(data, eps=0.5, min_samples=3):
    X = prepare(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    n_clusters = len(set(db.labels_) - {-1})
    n_noise = list(db.labels_).count(-1)
    print(f"=== DBSCAN Results (eps={eps}, min_samples={min_samples}) ===")
    print(f"Number of Clusters: {n_clusters}")
    print(f"Noise / Outlier Points (-1): {n_noise}")
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_, cmap='viridis')
    plt.title(f"DBSCAN: {n_clusters} Clusters, {n_noise} Outliers")
    plt.show()
    return db


# ------------------------------------------------------------------------------
# 7. UNIVERSAL PCA SOLVER (2D & 3D Projections)
# ------------------------------------------------------------------------------
def do_pca(data, n_components=2, colors=None):
    X = prepare(data)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"=== PCA Reduction ({X.shape[1]}D -> {n_components}D) ===")
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    if n_components == 2:
        plt.figure(figsize=(6, 4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis')
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("PCA 2D Projection")
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, cmap='viridis')
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        plt.title("PCA 3D Projection")
        plt.show()
    return X_pca


# ------------------------------------------------------------------------------
# 8. UNIVERSAL GMM (SOFT CLUSTERING) SOLVER
# ------------------------------------------------------------------------------
def do_gmm(data, k=3, cov='full'):
    X = prepare(data)
    gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42).fit(X)
    print(f"=== GMM Clustering (Components={k}, Covariance={cov}) ===")
    print("Hard Cluster Labels:\n", gmm.predict(X))
    print("\nSoft Probabilities (First 3 Data Points):\n", gmm.predict_proba(X)[:3])
    return gmm


# ------------------------------------------------------------------------------
# 9. UNIVERSAL CLUSTER EVALUATION METRICS (All Scores Together)
# ------------------------------------------------------------------------------
def all_metrics(data, labels):
    X = prepare(data)
    print("=== Cluster Quality Validation Metrics ===")
    print(f"1. Silhouette Score (Higher is better, max 1.0): {silhouette_score(X, labels):.4f}")
    print(f"2. Davies-Bouldin Index (Lower is better, min 0.0): {davies_bouldin_score(X, labels):.4f}")
    print(f"3. Calinski-Harabasz Score (Higher is better): {calinski_harabasz_score(X, labels):.4f}")
