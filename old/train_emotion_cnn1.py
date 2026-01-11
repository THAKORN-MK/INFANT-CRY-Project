import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1. Extract Mel-Spectrogram
# -----------------------------
def extract_mel(file_path):
    y, sr = librosa.load(file_path, duration=3)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# -----------------------------
# 2. Load Dataset
# -----------------------------
dataset_path = r"D:/EMOTDD/PRO/dataset"

classes = {
    "belly_pain": 0,
    "burping": 1,
    "cold_hot": 2
}

X = []
y = []

for label_name, label_id in classes.items():
    folder = os.path.join(dataset_path, label_name)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            mel = extract_mel(file_path)
            X.append(mel)
            y.append(label_id)

X = np.array(X)
X = X[..., np.newaxis]   # (samples, H, W, 1)
y = to_categorical(y, num_classes=3)

print("Total samples:", X.shape[0])

# -----------------------------
# 3. Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# 4. CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(3, activation='softmax')
])

# -----------------------------
# 5. Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 6. Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 7. Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# -----------------------------
# 8. Save Model
# -----------------------------
model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")
