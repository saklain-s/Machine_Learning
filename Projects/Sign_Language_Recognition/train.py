import numpy as np
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load preprocessed data
with np.load('data_preprocessed.npz') as data:
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

model = build_model()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('sign_cnn_model.h5', save_best_only=True)
]

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

print('Model trained and saved as sign_cnn_model.h5')
