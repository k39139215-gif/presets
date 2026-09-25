# Practical 3: Matplotlib Scatter Plot & Sine Wave
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
plt.scatter(iris.data[:, 0], iris.data[:, 2])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal vs Petal Length")
plt.show()

x = np.arange(0, 5, 0.1)
plt.plot(x, np.sin(x))
plt.title("Sine Wave")
plt.show()
