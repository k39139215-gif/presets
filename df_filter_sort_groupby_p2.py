# Practical 2: Student DataFrame Operations (Filter, Sort, GroupBy)
import pandas as pd

data = {
    "Student": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Department": ["CS", "IT", "CS", "IT", "CS"],
    "Marks": [85, 72, 90, 68, 88],
    "Age": [20, 21, 20, 22, 21]
}
df = pd.DataFrame(data)

print("--- Marks > 80 ---\n", df[df["Marks"] > 80])
print("\n--- Sorted by Marks ---\n", df.sort_values(by="Marks", ascending=False))
print("\n--- Dept Average Marks ---\n", df.groupby("Department")["Marks"].mean())
print("\n--- Dept Statistics ---\n", df.groupby("Department").agg({"Marks": ["mean", "max", "min"], "Age": "mean"}))
