import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Data collection and Data Preprocessing
sonar_data = pd.read_csv('Projects/Rock_Or_Mine/sonar.csv', header=None)
X = sonar_data.drop(columns=[60], axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the trained model to the backend prediction directory
model_path = os.path.join('Projects', 'Rock_Or_Mine', 'rockormine_backend', 'prediction', 'logistic_model.joblib')
joblib.dump(model, model_path)

# Evaluate the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(f"Training accuracy: {training_data_accuracy}")
print(f"Test accuracy: {test_data_accuracy}")

# Sample prediction
input_data = (0.0203,0.0121,0.0380,0.0128,0.0537,0.0874,0.1021,0.0852,0.1136,0.1747,0.2198,0.2721,0.2105,0.1727,0.2040,0.1786,0.1318,0.2260,0.2358,0.3107,0.3906,0.3631,0.4809,0.6531,0.7812,0.8395,0.9180,0.9769,0.8937,0.7022,0.6500,0.5069,0.3903,0.3009,0.1565,0.0985,0.2200,0.2243,0.2736,0.2152,0.2438,0.3154,0.2112,0.0991,0.0594,0.1940,0.1937,0.1082,0.0336,0.0177,0.0209,0.0134,0.0094,0.0047,0.0045,0.0042,0.0028,0.0036,0.0013,0.0016)
input_data_num = np.asarray(input_data)
input_data_reshape = input_data_num.reshape(1, -1)
prediction = model.predict(input_data_reshape)
if prediction[0] == 'R':
    print("Object is a Rock")
else:
    print("It is a mine")
