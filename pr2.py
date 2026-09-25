# ==============================================================================
# PRACTICAL 2: DATA MANIPULATION USING PANDAS (FILTER, SORT, GROUPBY)
# ==============================================================================
import pandas as pd

# Dataset
data = {
    'Roll_No': [1, 2, 3, 4, 5],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha', 'Karan'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Marks': [85, 72, 90, 68, 88],
    'Attendance': [90, 80, 88, 75, 92]
}
df = pd.DataFrame(data)
print("--- Original DataFrame ---\n", df)

# 1. Display only Name and Marks
print("\n1. Name and Marks only:\n", df[['Name', 'Marks']])

# 2. Display students scoring more than 75 marks
print("\n2. Marks > 75:\n", df[df['Marks'] > 75])

# 3. Sort students based on Marks (Descending)
print("\n3. Sorted by Marks (Descending):\n", df.sort_values(by='Marks', ascending=False))

# 4. Department-wise Summary (Average, Max Marks & Count)
print("\n4. Department-wise Summary:\n", df.groupby('Department').agg(
    Avg_Marks=('Marks', 'mean'),
    Highest_Marks=('Marks', 'max'),
    Student_Count=('Roll_No', 'count')
))

# 5. Display students with attendance > 85%
print("\n5. Attendance > 85%:\n", df[df['Attendance'] > 85])
