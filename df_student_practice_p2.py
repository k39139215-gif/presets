# Practical 2: Practice Question Solution (Complete)
import pandas as pd

data = {
    'Roll_No': [1, 2, 3, 4, 5],
    'Name': ['Amit', 'Priya', 'Rahul', 'Sneha', 'Karan'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
    'Marks': [85, 72, 90, 68, 88],
    'Attendance': [90, 80, 88, 75, 92]
}
df = pd.DataFrame(data)

print("1. Name and Marks:\n", df[['Name', 'Marks']])
print("\n2. Marks > 75:\n", df[df['Marks'] > 75])
print("\n3. Sorted by Marks (Descending):\n", df.sort_values(by='Marks', ascending=False))
print("\n4. Dept-wise Summary:\n", df.groupby('Department').agg(Avg_Marks=('Marks', 'mean'), Max_Marks=('Marks', 'max'), Count=('Roll_No', 'count')))
print("\n5. Attendance > 85%:\n", df[df['Attendance'] > 85])
