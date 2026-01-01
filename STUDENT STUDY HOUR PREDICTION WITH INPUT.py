import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


train_hours = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
train_marks = np.array([35,40,45,50,55,60,65,70,75,80,90,94,97,95,99])

model = LinearRegression()
model.fit(train_hours, train_marks)

user_hours = []

print("Enter study hours for 10 students:")
for i in range(15):
    h = float(input(f"Student {i+1} hours: "))
    user_hours.append(h)

user_hours = np.array(user_hours).reshape(-1,1)

predicted_marks = model.predict(user_hours)

students = np.arange(1, 16)

plt.bar(students, predicted_marks)
plt.xlabel("Students")
plt.ylabel("Predicted Marks")
plt.title("Student Marks Prediction based on Study Hours")
plt.show()
print("If a student studies", user_hours,
      "hours, predicted marks are:", round(predicted_marks[0], 2))
