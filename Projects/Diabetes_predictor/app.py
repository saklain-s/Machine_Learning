from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Data Preprocessing
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardizing data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train SVM Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting data from form
        input_data = [float(request.form[key]) for key in request.form]

        # Convert input into numpy array
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

        # Standardize input
        std_data = scaler.transform(input_data_as_numpy_array)

        # Make Prediction
        prediction = classifier.predict(std_data)

        result = "The person is Diabetic" if prediction[0] == 1 else "The person is Not Diabetic"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
