import numpy as np
import librosa
import tensorflow as tf

# -----------------------------
# โหลดโมเดล
# -----------------------------
model = tf.keras.models.load_model("belly_pain_model.h5")
print("Model loaded")

# -----------------------------
# แปลงเสียง -> Mel Spectrogram
# (ต้องเหมือนตอน train)
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
# ใส่ไฟล์เสียงทดสอบ
# -----------------------------
test_audio = r"D:/EMOTDD/PRO/test_audio.wav"  # เปลี่ยนเป็นไฟล์ของคุณ

mel = extract_mel(test_audio)
mel = mel[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(mel)[0][0]

print("Belly pain probability:", pred)

# -----------------------------
# ตีความผล
# -----------------------------
if pred >= 0.5:
    print("🔴 Result: Belly Pain")
else:
    print("🟢 Result: Not Belly Pain")
