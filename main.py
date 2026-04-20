import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Generate dataset (1000 rows)
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "age": np.random.randint(30, 80, n),
    "cholesterol": np.random.randint(150, 300, n),
    "bp": np.random.randint(100, 180, n),
    "heart_rate": np.random.randint(60, 110, n)
})

#Risk logic
data["risk"] = (
    (data["age"] > 50) |
    (data["cholesterol"] > 240) |
    (data["bp"] > 150)
).astype(int)

#Save dataset
data.to_csv("data.csv", index=False)

print("Dataset created with", len(data), "rows")

#Analysis
print("\nAverages:")
print(data.mean())

#Graphs
plt.hist(data["age"])
plt.title("Age Distribution")
plt.show()

plt.scatter(data["cholesterol"], data["bp"])
plt.title("Cholesterol vs BP")
plt.xlabel("Cholesterol")
plt.ylabel("BP")
plt.show()

#ML Model
X = data[["age", "cholesterol", "bp", "heart_rate"]]
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

#Prediction function
def predict_risk(age, cholesterol, bp, heart_rate):
    input_data = pd.DataFrame([[age, cholesterol, bp, heart_rate]],
                              columns=["age", "cholesterol", "bp", "heart_rate"])

    result = model.predict(input_data)
    return "High Risk" if result[0] == 1 else "Low Risk"

print("\nSample Prediction:", predict_risk(60, 260, 160, 90))
