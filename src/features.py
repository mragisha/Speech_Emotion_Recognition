"""Feature extraction: MFCC from raw audio files."""

import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import config


def extract_mfcc(filepath: str,
                 duration: float = config.SAMPLE_DURATION,
                 offset: float = config.SAMPLE_OFFSET,
                 n_mfcc: int = config.N_MFCC) -> np.ndarray:
    """Load a WAV and return a mean-pooled MFCC vector of shape (n_mfcc,)."""
    y, sr = librosa.load(filepath, duration=duration, offset=offset)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def build_feature_matrix(df: pd.DataFrame):
    """
    Extract MFCCs for every row in df and return (X, y, encoder).

    X : np.ndarray  shape (N, n_mfcc, 1)  — LSTM-ready input
    y : np.ndarray  shape (N, n_classes)  — one-hot labels
    enc : OneHotEncoder
    """
    X_raw = df["Path"].apply(extract_mfcc)
    X = np.array(list(X_raw))
    X = np.expand_dims(X, axis=-1)          # (N, 40, 1)

    enc = OneHotEncoder(sparse_output=False)
    y = enc.fit_transform(df[["Emotions"]])

    return X, y, enc
