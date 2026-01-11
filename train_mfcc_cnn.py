import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset"
CLASSES = ["belly_pain", "burping", "cold_hot"]
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 130   # time axis
EPOCHS = 60
BATCH_SIZE = 16

# -----------------------------
# MFCC extraction (2D)
# -----------------------------
def extract_mfcc(file_path):
    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Padding / Truncate
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# -----------------------------
# Load dataset
# -----------------------------
X, y = [], []

for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, cls)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            mfcc = extract_mfcc(path)
            X.append(mfcc)
            y.append(label)

X = np.array(X)
X = X[..., np.newaxis]   # (samples, 40, 130, 1)
y = tf.keras.utils.to_categorical(y, num_classes=3)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# CNN Model (2D)
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu",
           input_shape=(40,130,1)),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# -----------------------------
# Save
# -----------------------------
model.save("emotion_mfcc_cnn.h5")
print("Model saved: emotion_mfcc_cnn.h5")
