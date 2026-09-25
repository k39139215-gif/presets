# ==============================================================================
# ULT MASTER SEARCH & RETRIEVAL SCRIPT (PRACTICAL 1 TO 11)
# ==============================================================================

PRACTICALS = {
1: """# PRACTICAL 1: DATA PREPROCESSING & SCALING
# Keywords: SimpleImputer, KNNImputer, MinMaxScaler, StandardScaler, CountVectorizer, RobustScaler
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

data = np.array([[1, 2, np.nan], [4, np.nan, 3], [np.nan, 7, 6]])
print("Imputed:\\n", SimpleImputer(strategy='mean').fit_transform(data))
print("KNN Imputed:\\n", KNNImputer(n_neighbors=2).fit_transform(data))
print("MinMax:\\n", MinMaxScaler().fit_transform(data[:2, :2]))
print("Standard:\\n", StandardScaler().fit_transform([[10, 20], [15, 30], [25, 45]]))
cv = CountVectorizer()
print("Vocab:", cv.fit(["bad service", "good service"]).vocabulary_)
""",

2: """# PRACTICAL 2: PANDAS (FILTER, SORT, GROUPBY)
# Keywords: filter, sort, groupby, agg, DataFrame, Department, Marks, Attendance
import pandas as pd

data = {
    'Roll_No': [1, 2, 3, 4, 5],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha', 'Karan'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Marks': [85, 72, 90, 68, 88],
    'Attendance': [90, 80, 88, 75, 92]
}
df = pd.DataFrame(data)
print(df[['Name', 'Marks']])                                      # Display specific columns
print(df[df['Marks'] > 75])                                       # Filter > 75
print(df.sort_values(by='Marks', ascending=False))               # Sort descending
print(df.groupby('Department').agg(Avg=('Marks', 'mean'), Max=('Marks', 'max'), Count=('Roll_No', 'count')))
print(df[df['Attendance'] > 85])                                 # Attendance > 85%
""",

3: """# PRACTICAL 3: PYTHON LIBRARIES FOR UNSUPERVISED LEARNING
# Keywords: NumPy, Pandas, Matplotlib, Seaborn, pairplot, heatmap, corr, load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
print("NumPy Mean per column:", np.mean(df.values, axis=0))

plt.scatter(df.iloc[:, 0], df.iloc[:, 2])
plt.xlabel("Sepal Length"); plt.ylabel("Petal Length"); plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm"); plt.show()
print("Standardized:\\n", StandardScaler().fit_transform(df)[:3])
""",

4: """# PRACTICAL 4: K-MEANS CLUSTERING
# Keywords: KMeans, n_clusters, cluster_centers_, centroids, Wine, Iris, predict
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Iris K=3
X_iris, _ = load_iris(return_X_y=True)
km = KMeans(n_clusters=3, random_state=42).fit(X_iris)
print("Iris Labels:", km.labels_)

# Wine K=2, 4, 5 comparison
X_wine = StandardScaler().fit_transform(load_wine().data)
for k in [2, 4, 5]:
    km_w = KMeans(n_clusters=k, random_state=42).fit(X_wine)
    print(f"Wine K={k} Sizes:", pd.Series(km_w.labels_).value_counts().to_dict())
""",

5: """# PRACTICAL 5: ELBOW METHOD FOR OPTIMAL K
# Keywords: Elbow, WCSS, SSE, inertia_, optimal K, make_blobs, penguins
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)

wcss = [KMeans(n_clusters=k, random_state=42).fit(df).inertia_ for k in range(1, len(df)+1)]

plt.plot(range(1, len(df)+1), wcss, marker='o')
plt.xlabel("K"); plt.ylabel("WCSS"); plt.title("Elbow Method"); plt.show()

# Optimal K = 4
df['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(df[['Income', 'Spending']])
print(df)
""",

6: """# PRACTICAL 6: SILHOUETTE SCORE METHOD
# Keywords: Silhouette, silhouette_score, silhouette_samples, argmax, best K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(load_iris().data)
scores = [silhouette_score(X, KMeans(n_clusters=k, random_state=42).fit_predict(X)) for k in range(2, 11)]

for k, s in zip(range(2, 11), scores):
    print(f"K={k}, Score={s:.4f}")

best_k = range(2, 11)[np.argmax(scores)]
print(f"Optimal K = {best_k}")

plt.plot(range(2, 11), scores, marker='o')
plt.xlabel("K"); plt.ylabel("Silhouette Score"); plt.show()
""",

7: """# PRACTICAL 7: HIERARCHICAL CLUSTERING & DENDROGRAM
# Keywords: Hierarchical, Agglomerative, Dendrogram, linkage, ward, euclidean, tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    "Income": [15, 16, 17, 18, 20, 22, 60, 62, 65, 68, 90, 95],
    "Spending": [39, 81, 6, 77, 40, 76, 55, 52, 59, 61, 20, 18]
}
df = pd.DataFrame(data)
X = StandardScaler().fit_transform(df)

dendrogram(linkage(X, method='ward'), labels=df.index + 1)
plt.title("Dendrogram"); plt.show()

df['Cluster'] = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)
print(df)
""",

8: """# PRACTICAL 8: DBSCAN CLUSTERING & OUTLIERS
# Keywords: DBSCAN, eps, min_samples, noise, outliers, -1, density
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)

n_clusters = len(set(labels) - {-1})
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Outliers/Noise: {n_noise}")

if n_clusters > 1:
    print("Silhouette Score:", round(silhouette_score(X, labels), 4))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("DBSCAN"); plt.show()
""",

9: """# PRACTICAL 9: PCA FOR DIMENSION REDUCTION
# Keywords: PCA, Dimension Reduction, explained_variance_ratio_, Eigenvalues, Wine 13D to 2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", wine.data.shape)
print("Reduced shape:", X_pca.shape)
print("Total Variance Retained:", round(np.sum(pca.explained_variance_ratio_) * 100, 2), "%")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine.target, cmap='viridis')
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Wine PCA"); plt.show()
""",

10: """# PRACTICAL 10: PCA FOR VISUALIZATION
# Keywords: PCA Visualization, 2D projection, Iris species plot, color legend
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
X_pca = PCA(n_components=2).fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Species'] = iris.target

for t, col, name in zip([0, 1, 2], ['r', 'g', 'b'], iris.target_names):
    sub = pca_df[pca_df['Species'] == t]
    plt.scatter(sub['PC1'], sub['PC2'], c=col, label=name)

plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Iris 2D PCA"); plt.legend(); plt.show()
""",

11: """# PRACTICAL 11: GAUSSIAN MIXTURE MODEL (GMM)
# Keywords: GMM, Gaussian, GaussianMixture, soft clustering, predict_proba, covariance_type, EM
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X, _ = load_iris(return_X_y=True)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(X)
print("Hard Labels:", gmm.predict(X)[:5])
print("Soft Probabilities:\\n", gmm.predict_proba(X)[0])

for cov in ['full', 'tied', 'diag', 'spherical']:
    lbl = GaussianMixture(n_components=3, covariance_type=cov, random_state=42).fit_predict(X)
    print(f"Covariance {cov:9s} Score: {silhouette_score(X, lbl):.4f}")
"""
}

def find(word):
    """Keyword dhoondh kar batata hai ki kaun se practical me aayega"""
    word = word.lower()
    matches = []
    for num, code in PRACTICALS.items():
        if word in code.lower():
            # Extract title line
            first_line = [l for l in code.splitlines() if l.startswith("# PRACTICAL")][0]
            matches.append((num, first_line))
    
    if matches:
        print(f"\n🔍 Keyword '{word}' found in:")
        for num, title in matches:
            print(f"   👉 Practical {num} -> {title}")
        print(f"\n💡 Run: %load /content/drive/MyDrive/YOUR_FOLDER/pr{matches[0][0]}.py\n")
    else:
        print(f"❌ No matching practical found for '{word}'.")

def show(n):
    """Practical number dekar poora code screen par print karta hai"""
    if n in PRACTICALS:
        print(PRACTICALS[n])
    else:
        print(f"Practical {n} not found! Available: 1 to 11")
