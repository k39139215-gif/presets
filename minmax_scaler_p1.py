# Practical 1: MinMaxScaler (Scale to 0 to 1)
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original Data:\n", data)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print("\nMinMax Scaled Data (0 to 1):\n", scaled_data)
