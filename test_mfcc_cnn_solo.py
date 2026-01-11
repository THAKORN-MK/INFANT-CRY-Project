import numpy as np
import librosa
import tensorflow as tf

MODEL_PATH = "emotion_mfcc_cnn.h5"
TEST_FILE = "test.wav"

CLASSES = ["Belly Pain", "Burping", "Cold or Hot"]
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 130

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

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
# Predict
# -----------------------------
mfcc = extract_mfcc(TEST_FILE)
mfcc = mfcc[np.newaxis, ..., np.newaxis]

pred = model.predict(mfcc)[0]

for cls, p in zip(CLASSES, pred):
    print(f"{cls}: {p:.2f}")

idx = np.argmax(pred)
print("\n🎯 Predicted Emotion:", CLASSES[idx])
print("Confidence:", pred[idx])
