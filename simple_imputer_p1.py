# Practical 1: Remove Null Values using SimpleImputer (Mean)
import numpy as np
from sklearn.impute import SimpleImputer

data = np.array([[1, 2, np.nan], [4, np.nan, 3], [np.nan, 7, 6]])
print("Original Data:\n", data)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_data = imputer.fit_transform(data)
print("\nImputed Data (Mean):\n", imputed_data)
