import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save_model():
    # Load sonar data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'sonar.csv')
    sonar_data = pd.read_csv(data_path, header=None)

    # Prepare data
    X = sonar_data.drop(columns=[60], axis=1)
    Y = sonar_data[60]

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Evaluate model
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)

    print(f"Training accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'logistic_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
