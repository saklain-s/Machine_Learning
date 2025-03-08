### Diabetes Prediction using Support Vector Machine (SVM)

## Overview

This project is a **Diabetes Prediction System** using Machine Learning techniques. It utilizes a **Support Vector Machine (SVM) Classifier** to predict whether a person is diabetic based on medical features. The dataset used is the **Pima Indians Diabetes Dataset**.

## Dataset

The dataset is read from a CSV file (`diabetes.csv`) and contains multiple medical diagnostic features such as:

- **Pregnancies**
- **Glucose level**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin Level**
- **Body Mass Index (BMI)**
- **Diabetes Pedigree Function**
- **Age**
- **Outcome** (1 = Diabetic, 0 = Not Diabetic)

## Requirements

To run this project, install the required Python libraries:

```shell
pip install numpy pandas scikit-learn


```

## Steps in the Model
1. Load the dataset using pandas.read_csv().
2. Explore the dataset, including: Checking the shape, Getting a statistical summary, Analyzing the class distribution.
3. Preprocess the data:
Separate features (X) and labels (Y).
Standardize features using StandardScaler().
Split the data into training and testing sets (80% train, 20% test).
Train an SVM model with a linear kernel.
Evaluate model accuracy on both training and test data.
