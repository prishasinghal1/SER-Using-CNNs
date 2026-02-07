import sys
import os
import numpy as np
import librosa
import tensorflow as tf

EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]

MODEL_PATH = "ser_cnn_model.h5"


def preprocess_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)

    max_len = 3 * sr
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < 128:
        log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :128]

    log_mel = (log_mel - log_mel.mean()) / log_mel.std()

  
    return log_mel[np.newaxis, ..., np.newaxis]


def main():
    args = [a for a in sys.argv[1:] if a.endswith(".wav")]

    if len(args) == 0:
        print("ERROR: No .wav file provided.")
        print("Usage: python predict.py <audio_file.wav>")
        sys.exit(1)

    audio_path = args[0]

    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)

    features = preprocess_audio(audio_path)
    predictions = model.predict(features)[0]

    idx = np.argmax(predictions)
    print(f"Predicted Emotion: {EMOTIONS[idx]}")
    print(f"Confidence: {predictions[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
