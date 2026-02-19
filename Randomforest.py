import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Example dataset (marks vs salary/score/etc.)
X = np.array([[10],[20],[30],[40],[50],[60],[70]])
y = np.array([100,200,300,400,500,600,700])

# Create model
RandomForestRegModel = RandomForestRegressor()

# Train model
RandomForestRegModel.fit(X, y)

# Predict for 70 marks
X_marks = np.array([[70]])
print(RandomForestRegModel.predict(X_marks))
