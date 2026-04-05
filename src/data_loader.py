"""Load and preprocess the TESS dataset into a DataFrame."""

import os
import pandas as pd
import config


def load_tess(tess_path: str = config.TESS_PATH) -> pd.DataFrame:
    """
    Walk the TESS directory and return a DataFrame with columns:
    ['Emotions', 'Path'].

    File naming convention: <speaker>_<word>_<emotion>.wav
    'ps' is remapped to 'surprise'.
    """
    file_emotion, file_path = [], []

    for speaker_dir in os.listdir(tess_path):
        speaker_full = os.path.join(tess_path, speaker_dir)
        if not os.path.isdir(speaker_full):
            continue
        for wav_file in os.listdir(speaker_full):
            stem = wav_file.split(".")[0]
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            emotion = parts[2].lower()
            if emotion == "ps":
                emotion = "surprise"
            file_emotion.append(emotion)
            file_path.append(os.path.join(speaker_full, wav_file))

    df = pd.DataFrame({"Emotions": file_emotion, "Path": file_path})
    return df
