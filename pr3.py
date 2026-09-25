# ==============================================================================
# PRACTICAL 3: PYTHON LIBRARIES FOR UNSUPERVISED LEARNING
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Load Iris Dataset in Pandas
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("--- Iris Data Head ---\n", df.head())
print("\n--- Summary Statistics ---\n", df.describe())

# 2. NumPy Operations
arr = df.values
print("Shape:", arr.shape)
print("Mean per feature:", np.mean(arr, axis=0))
print("Max per feature:", np.max(arr, axis=0))

# 3. Matplotlib Scatter Plot
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"])
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Iris Sepal vs Petal Length")
plt.show()

# 4. Seaborn Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Iris Feature Correlation Heatmap")
plt.show()

# 5. Scikit-learn StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("\nStandardized Features (First 3 rows):\n", scaled_data[:3])
