import numpy as np
import librosa
import tensorflow as tf

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("emotion_model.h5")
print("Model loaded")

# -----------------------------
# Label names
# -----------------------------
labels = ["Belly Pain", "Burping", "Cold or Hot"]

# -----------------------------
# Extract Mel
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
# Test audio
# -----------------------------
test_audio = r"D:/EMOTDD/PRO/test_audio.wav"  # เปลี่ยนไฟล์เอง

mel = extract_mel(test_audio)
mel = mel[np.newaxis, ..., np.newaxis]

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(mel)[0]

for i, score in enumerate(pred):
    print(f"{labels[i]}: {score:.2f}")

result = labels[np.argmax(pred)]
print("\n🎯 Predicted Emotion:", result)
