# Practical 2: Employee Summary with Filter Items & Named Aggregations
import pandas as pd

data = {
    "Emp_ID": [101, 102, 103, 104, 105, 106],
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Karan", "Neha"],
    "Department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
    "Salary": [55000, 42000, 68000, 50000, 47000, 72000],
    "Experience": [3, 2, 5, 4, 1, 6]
}
df = pd.DataFrame(data)

print("Selected Columns:\n", df.filter(items=["Name", "Department", "Salary"]))
print("\nSalary > 50000:\n", df[df["Salary"] > 50000])

summary = df.groupby("Department").agg(
    Avg_Salary=("Salary", "mean"),
    Max_Salary=("Salary", "max"),
    Count=("Emp_ID", "count")
)
print("\nDepartment Summary:\n", summary)
