# ==============================================================================
# PRACTICAL 1: DATA PREPROCESSING AND SCALING
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer

# 1. SimpleImputer (Mean strategy)
data = np.array([[1, 2, np.nan], [4, np.nan, 3], [np.nan, 7, 6]])
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
print("--- Simple Imputed Data ---\n", imputer.fit_transform(data))

# 2. KNNImputer (Neighbor-based imputation)
data_knn = np.array([[1, 2, np.nan], [5, 1, 6], [4, np.nan, 3], [9, 7, 6]])
knn = KNNImputer(n_neighbors=2)
print("--- KNN Imputed Data ---\n", knn.fit_transform(data_knn))

# 3. MinMaxScaler (0 to 1 range)
scaler_mm = MinMaxScaler()
print("--- MinMax Scaled Data ---\n", scaler_mm.fit_transform(data[:2, :2]))

# 4. CountVectorizer (Text to Bag of Words)
docs = ["Your Service is very bad", "TCS is service based company"]
cv = CountVectorizer()
print("--- Word Counts ---\n", cv.fit_transform(docs).toarray())
print("Vocabulary:", cv.vocabulary_)

# 5. StandardScaler (Mean=0, Std=1)
scaler_std = StandardScaler()
print("--- Standard Scaled Data ---\n", scaler_std.fit_transform([[10, 20], [15, 30], [25, 45]]))
