# Practical 1: RobustScaler (Outlier Resistant Scaling)
import numpy as np
from sklearn.preprocessing import RobustScaler

data = np.array([[1, 2], [2, 3], [3, 2], [100, 200]])
print("Original Data:\n", data)

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)
print("\nRobust Scaled Data:\n", scaled_data)
