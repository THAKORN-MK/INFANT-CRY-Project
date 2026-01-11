import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. แปลงเสียง -> Mel-Spectrogram
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
# 2. โหลดข้อมูล Belly pain
# -----------------------------
X = []
dataset_path = "dataset/belly_pain"

for file in os.listdir(dataset_path):
    if file.endswith(".wav"):
        file_path = os.path.join(dataset_path, file)
        mel = extract_mel(file_path)
        X.append(mel)

X = np.array(X)
X = X[..., np.newaxis]  # (samples, height, width, 1)

print("Total Belly pain samples:", X.shape[0])

# -----------------------------
# 3. Label (คลาสเดียว = 0)
# -----------------------------
y = np.zeros(len(X))  # belly pain = 0

# -----------------------------
# 4. แบ่ง Train / Test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. สร้าง CNN
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
    Dense(64, activation='relu'),
    Dropout(0.4),

    Dense(1, activation='sigmoid')  # binary output
])

# -----------------------------
# 6. Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 7. Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 8. Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# -----------------------------
# 9. Save model
# -----------------------------
model.save("belly_pain_model.h5")
print("Model saved as belly_pain_model.h5")
