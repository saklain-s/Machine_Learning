DATASET_PATH = '/home/codespace/.cache/kagglehub/datasets/datamunge/sign-language-mnist/versions/1/sign_mnist_train.csv'  # Updated path to downloaded dataset
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Example for Sign Language MNIST CSV dataset
DATASET_PATH = '/home/codespace/.cache/kagglehub/datasets/datamunge/sign-language-mnist/versions/1/sign_mnist_train.csv'  # Updated dataset path
IMG_SIZE = 64

# Load dataset
if DATASET_PATH.endswith('.csv'):
    df = pd.read_csv(DATASET_PATH)
    labels = df['label'].values
    images = df.drop('label', axis=1).values.reshape(-1, 28, 28)
else:
    raise NotImplementedError('Only CSV dataset supported in this starter script.')


# Reshape each image from flat (784,) to (28,28) before resizing
X = []
for i, img in enumerate(images):
    try:
        img = img.astype(np.uint8)  # Ensure correct type
        img_2d = img.reshape(28, 28)
        img_resized = cv2.resize(img_2d, (IMG_SIZE, IMG_SIZE))
        X.append(img_resized)
    except Exception as e:
        print(f"Error processing image {i}: {e}")
X = np.array(X)
X = X.astype('float32') / 255.0
X = np.expand_dims(X, -1)

y = to_categorical(labels)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

np.savez('data_preprocessed.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
print('Data preprocessed and saved to data_preprocessed.npz')
