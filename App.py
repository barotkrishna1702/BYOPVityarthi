import pandas as pd
from sklearn.linear_model import LinearRegression
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "sleep_hours": [8, 7, 6, 6, 7, 5, 6, 5],
    "attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "marks": [35, 40, 50, 55, 65, 70, 80, 90]
}
df = pd.DataFrame(data)

X = df[["study_hours", "sleep_hours", "attendance"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)

print("\nEnter student details:")

study = float(input("Study hours per day: "))
sleep = float(input("Sleep hours per day: "))
attendance = float(input("Attendance percentage: "))

prediction = model.predict([[study, sleep, attendance]])
print(f"\nPredicted Marks: {prediction[0]:.2f}")
