# Practical 1: Remove Null Values using KNNImputer
import numpy as np
from sklearn.impute import KNNImputer

data = np.array([[1, 2, np.nan], [5, 1, 6], [4, np.nan, 3], [9, 7, 6]])
print("Original Data:\n", data)

knn_imputer = KNNImputer(n_neighbors=2)
imputed_data = knn_imputer.fit_transform(data)
print("\nImputed Data (KNN, k=2):\n", imputed_data)
