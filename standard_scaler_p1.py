# Practical 1: StandardScaler (Mean=0, Std=1)
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([[10, 20], [15, 30], [25, 45], [30, 60]])
print("Original Data:\n", data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("\nStandard Scaled Data (Z-score):\n", scaled_data)
