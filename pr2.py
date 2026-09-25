# ==============================================================================
# Unsupervised Learning Technique Journal - Practical No 2
# Aim: To perform data manipulation and transformation on a DataFrame using
#      Pandas functions such as filter, sort, and groupby.
# ==============================================================================
import pandas as pd

# ------------------------------------------------------------------------------
# CODE: 1 (Student Dataset Operations)
# ------------------------------------------------------------------------------
print("=================== CODE 1: STUDENT DATA ===================")
data1 = {
    "Student": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Department": ["CS", "IT", "CS", "IT", "CS"],
    "Marks": [85, 72, 90, 68, 88],
    "Age": [20, 21, 20, 22, 21]
}
df1 = pd.DataFrame(data1)
print("Original DataFrame:\n", df1)

# Filter rows
print("\nStudents with Marks greater than 80:")
filtered = df1[df1["Marks"] > 80]
print(filtered)

# Sort DataFrame
print("\nSorted by Marks (Descending):")
sorted_df = df1.sort_values(by="Marks", ascending=False)
print(sorted_df)

# Group DataFrame
print("\nAverage Marks by Department:")
grouped = df1.groupby("Department")["Marks"].mean()
print(grouped)

# Multiple Aggregations
print("\nDepartment-wise Statistics:")
stats = df1.groupby("Department").agg({
    "Marks": ["mean", "max", "min"],
    "Age": "mean"
})
print(stats)


# ------------------------------------------------------------------------------
# CODE: 2 (Employee Dataset Operations)
# ------------------------------------------------------------------------------
print("\n=================== CODE 2: EMPLOYEE DATA ===================")
data2 = {
    "Emp_ID": [101, 102, 103, 104, 105, 106],
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Karan", "Neha"],
    "Department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
    "Salary": [55000, 42000, 68000, 50000, 47000, 72000],
    "Experience": [3, 2, 5, 4, 1, 6]
}
df2 = pd.DataFrame(data2)
print("Original DataFrame:\n", df2)

# Filter selected columns
print("\nSelected Columns:")
print(df2.filter(items=["Name", "Department", "Salary"]))

# Filter rows
print("\nEmployees having Salary > 50000:")
print(df2[df2["Salary"] > 50000])

# Sort DataFrame
print("\nSorted by Salary (Descending):")
print(df2.sort_values(by="Salary", ascending=False))

# GroupBy Operation
print("\nDepartment-wise Summary:")
result = df2.groupby("Department").agg(
    Average_Salary=("Salary", "mean"),
    Maximum_Salary=("Salary", "max"),
    Employee_Count=("Emp_ID", "count")
)
print(result)


# ------------------------------------------------------------------------------
# PRACTICE QUESTION (Complete Solution)
# ------------------------------------------------------------------------------
print("\n=================== PRACTICE QUESTION ===================")
data_pq = {
    'Roll_No': [1, 2, 3, 4, 5],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha', 'Karan'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Marks': [85, 72, 90, 68, 88],
    'Attendance': [90, 80, 88, 75, 92]
}
df_pq = pd.DataFrame(data_pq)

# 1. Display only Name and Marks columns
print("\n1. Name and Marks only:\n", df_pq[['Name', 'Marks']])

# 2. Display students scoring more than 75 marks
print("\n2. Students scoring > 75 marks:\n", df_pq[df_pq['Marks'] > 75])

# 3. Sort students based on Marks in descending order
print("\n3. Sorted by Marks (Descending):\n", df_pq.sort_values(by='Marks', ascending=False))

# 4. Find department-wise: Average Marks, Highest Marks, Number of Students
print("\n4. Department-wise Summary:\n", df_pq.groupby('Department').agg(
    Average_Marks=('Marks', 'mean'),
    Highest_Marks=('Marks', 'max'),
    Student_Count=('Roll_No', 'count')
))

# 5. Display students with attendance greater than 85%
print("\n5. Attendance > 85%:\n", df_pq[df_pq['Attendance'] > 85])
