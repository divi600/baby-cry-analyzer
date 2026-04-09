import librosa
import numpy as np
import joblib
import io
import soundfile as sf

# Load saved scaler & encoder
scaler = joblib.load("backend/scaler.pkl")
le = joblib.load("backend/label_encoder.pkl")


def extract_features(audio_bytes):
    # Convert bytes → audio signal
    y, sr = sf.read(io.BytesIO(audio_bytes))
    
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # convert stereo to mono

    # ===== SAME FEATURES AS TRAINING =====
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs_delta = np.mean(mfccs_delta.T, axis=0)

    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

    features = np.hstack([
        mfccs, mfccs_delta, chroma, mel,
        contrast, tonnetz, zcr, rms,
        centroid, bandwidth, rolloff
    ])

    # Scale
    features = scaler.transform([features])

    # Add time dimension (same as training)
    features = np.expand_dims(features, axis=1)

    return features
