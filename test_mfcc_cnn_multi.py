import os
import numpy as np
import librosa
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "emotion_mfcc_cnn.h5"
TEST_FOLDER = "test_audio"

CLASSES = ["Belly Pain", "Burping", "Cold or Hot"]
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 130

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded\n")

# -----------------------------
# MFCC extraction
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

    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# -----------------------------
# Test all wav files
# -----------------------------
files = sorted([f for f in os.listdir(TEST_FOLDER) if f.endswith(".wav")])

for i, file in enumerate(files, 1):
    path = os.path.join(TEST_FOLDER, file)

    mfcc = extract_mfcc(path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc, verbose=0)[0]
    idx = np.argmax(pred)

    print(f"🎵 Audio {i}: {file}")
    for cls, p in zip(CLASSES, pred):
        print(f"   {cls}: {p:.2f}")

    print(f"➡️  Predicted: {CLASSES[idx]} (Confidence {pred[idx]:.2f})")
    print("-" * 40)
