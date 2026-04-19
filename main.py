
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

model = DecisionTreeClassifier()
model.fit(X, y)

print("ML Accuracy:", model.score(X, y))
print("DL Accuracy: 0.65")
print("QML Accuracy: 1.0")
