# Practical 3: Load Iris Dataset & Explore Properties
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Head:\n", df.head())
print("\nInfo:")
df.info()
print("\nSummary Description:\n", df.describe())
print("\nNull Values Count:\n", df.isnull().sum())
