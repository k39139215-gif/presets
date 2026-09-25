# ==============================================================================
# Unsupervised Learning Technique Journal - Practical No 1
# Aim: To study, preprocess and visualise the data
# ==============================================================================

# ------------------------------------------------------------------------------
# CODE: 1 Remove null values (NAN) using SimpleImputer
# ------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer
import numpy as np

data = np.array([[1, 2, np.nan], [4, np.nan, 3], [np.nan, 7, 6]])
print("---Original data-------------")
print(data)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

print("---Imputed data-------------")
imputed_data = imputer.fit_transform(data)
print(imputed_data)


# ------------------------------------------------------------------------------
# CODE: 2 Remove null values (NAN) using KNNImputer
# ------------------------------------------------------------------------------
from sklearn.impute import KNNImputer
import numpy as np

data = np.array([[1, 2, np.nan], [5, 1, 6], [4, np.nan, 3], [9, 7, 6]])
print("\n---Original data-------------")
print(data)

knn_imputer = KNNImputer(n_neighbors=2)
imputed_data = knn_imputer.fit_transform(data)

print("---Imputed data-------------")
print(imputed_data)


# ------------------------------------------------------------------------------
# CODE: 3 MinMaxScaler
# ------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
data_new = np.array([[10, 20], [15, 30], [25, 45], [30, 60]])

print("\n---Original data-------------")
print(data)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print("---Scaled data-------------")
print(scaled_data)

print("---Original data-------------")
print(data_new)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_new)
print("---Scaled data-------------")
print(scaled_data)


# ------------------------------------------------------------------------------
# CODE: 4 TEXT TO CATEGORICAL DATA
# ------------------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Your Service is very very bad",
    "TCS is service based company",
    "You work in bad service company"
]

counter_vectorizer = CountVectorizer()
count_matrix = counter_vectorizer.fit_transform(documents)

print("\nVocabulary: ", counter_vectorizer.vocabulary_)
print(count_matrix.toarray())


# ------------------------------------------------------------------------------
# CODE: 5 Dataset Preprocessing (California Housing / Scalers)
# ------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer

# Load California Housing Dataset (from file if uploaded, or fetch directly)
try:
    df = pd.read_csv('housing.csv')
except Exception:
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True).frame

df = df.select_dtypes(include=np.number)
print("\n--- Original Housing Data (Head) ---")
print(df.head())

# MinMaxScaler on Housing Data
scaler_mm = MinMaxScaler()
scaled_df_mm = pd.DataFrame(scaler_mm.fit_transform(df), columns=df.columns)
print("\n--- MinMaxScaler Result (Head) ---")
print(scaled_df_mm.head())

# StandardScaler on Housing Data
scaler_std = StandardScaler()
scaled_df_std = pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)
print("\n--- StandardScaler Result (Head) ---")
print(scaled_df_std.head())

# RobustScaler on Housing Data
scaler_robust = RobustScaler()
scaled_df_robust = pd.DataFrame(scaler_robust.fit_transform(df), columns=df.columns)
print("\n--- RobustScaler Result (Head) ---")
print(scaled_df_robust.head())

# Normalizer on Housing Data (with SimpleImputer first)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
original_columns = df.columns
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=original_columns)

scaler_norm = Normalizer()
scaled_df_norm = pd.DataFrame(scaler_norm.fit_transform(df_imputed), columns=original_columns)
print("\n--- Normalizer Result (Head) ---")
print(scaled_df_norm.head())
