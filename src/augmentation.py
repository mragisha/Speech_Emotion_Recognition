"""Audio augmentation utilities: noise injection, time-stretch, pitch-shift."""

import numpy as np
import librosa


def add_noise(data: np.ndarray) -> np.ndarray:
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])


def time_stretch(data: np.ndarray, rate: float = 0.8) -> np.ndarray:
    return librosa.effects.time_stretch(data, rate=rate)


def pitch_shift(data: np.ndarray, sr: int, n_steps: int = 2) -> np.ndarray:
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)


def get_all_augmentations(data: np.ndarray, sr: int) -> dict:
    """Return a dict of {label: augmented_signal} for all augmentation types."""
    return {
        "Original":        data,
        "Noise":           add_noise(data),
        "Stretch (0.8x)":  time_stretch(data, rate=0.8),
        "Pitch +2":        pitch_shift(data, sr, n_steps=2),
        "Pitch -2":        pitch_shift(data, sr, n_steps=-2),
    }
