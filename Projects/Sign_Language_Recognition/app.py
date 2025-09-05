from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sign_cnn_model.h5')
IMG_SIZE = 64

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Map class indices to labels (update as per your dataset)
CLASS_LABELS = [chr(i) for i in range(65, 91) if i != 74 and i != 90]  # A-Y (no J, Z for ASL MNIST)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            pred = model.predict(img)
            class_idx = np.argmax(pred)
            prediction = CLASS_LABELS[class_idx]
    return render_template('index.html', prediction=prediction)

# Live prediction endpoint for camera frames
@app.route('/live_predict', methods=['POST'])
def live_predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'prediction': None})
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred = model.predict(img)
    class_idx = int(np.argmax(pred))
    prediction = CLASS_LABELS[class_idx]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
