# Practical 3: NumPy Operations on DataFrame
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
data = df.values

print("Array Shape:", data.shape)
print("Mean per Column:", np.mean(data, axis=0))
print("Max per Column:", np.max(data, axis=0))
