# ==============================================================================
# Unsupervised Learning Technique Journal - Practical No 3
# Aim: Illustrate various python libraries for unsupervised learning techniques
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# 1. Import Required Libraries and Load Dataset
# ------------------------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
print("--- Dataset Head ---")
print(df.head())


# ------------------------------------------------------------------------------
# 2. Get Dataset information
# ------------------------------------------------------------------------------
print("\n--- Dataset Info ---")
df.info()
print("\n---------df description---------")
print(df.describe())
print("\n---------isnull count-----------")
print(df.isnull().sum())


# ------------------------------------------------------------------------------
# 3. NumPy Operations
# ------------------------------------------------------------------------------
data = df.values

print("\n--- NumPy Operations ---")
print("Shape:", data.shape)
print("Mean per feature:")
print(np.mean(data, axis=0))
print("Maximum per feature:")
print(np.max(data, axis=0))


# ------------------------------------------------------------------------------
# 4. Data Visualization using Matplotlib
# ------------------------------------------------------------------------------
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot")
plt.show()

# Sine wave plot
x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave Plot")
plt.show()


# ------------------------------------------------------------------------------
# 5. Data Visualization using Seaborn
# ------------------------------------------------------------------------------
# Iris Pairplot
sns.pairplot(df)
plt.show()

# Custom DataFrame Pairplot
df_custom = pd.DataFrame({
    'Height': [160, 165, 170, 175, 180],
    'Weight': [55, 60, 68, 75, 80],
    'Age': [20, 21, 22, 23, 24]
})
sns.pairplot(df_custom)
plt.show()


# ------------------------------------------------------------------------------
# 6. Correlation Heatmap
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------------------------
# 7. Standardize Dataset
# ------------------------------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("\n--- Standardized Data (First 5 Rows) ---")
print(scaled_data[:5])
