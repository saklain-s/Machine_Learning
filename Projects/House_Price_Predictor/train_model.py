import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset (using a sample dataset from sklearn)
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model in the project directory
import os
model_path = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
joblib.dump(model, model_path)
print(f'Model trained and saved as {model_path}')
