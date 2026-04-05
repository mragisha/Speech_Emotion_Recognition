"""
Entry point for Speech Emotion Recognition training.

Usage
-----
# Train all four models (default):
    python main.py

# Train specific models only:
    python main.py --models lstm bilstm

# Change dataset root at runtime:
    SER_DATA_ROOT=/path/to/data python main.py
"""

import argparse
import os

import warnings
warnings.filterwarnings("ignore")

import librosa
import config
from src.data_loader import load_tess
from src.features import build_feature_matrix, extract_mfcc
from src.augmentation import get_all_augmentations
from src.visualization import (
    plot_emotion_distribution,
    plot_waveforms,
    plot_mel_spectrograms,
    plot_mfcc_heatmap,
)
from src.models import MODEL_REGISTRY
from src.train import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip exploratory visualizations (faster)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print("Loading TESS dataset …")
    df = load_tess()
    df.to_csv(config.CSV_SAVE, index=False)
    print(f"  {len(df)} samples  |  emotions: {sorted(df['Emotions'].unique())}")

    # ── 2. Exploratory visualisations ─────────────────────────────────────────
    if not args.skip_viz:
        print("Generating exploratory plots …")
        plot_emotion_distribution(df)

        sample_path = df["Path"].iloc[0]
        sample_data, sample_sr = librosa.load(sample_path)
        augmentations = get_all_augmentations(sample_data, sample_sr)
        plot_waveforms(augmentations, sample_sr)
        plot_mel_spectrograms(augmentations, sample_sr)
        plot_mfcc_heatmap(df, extract_mfcc)

    # ── 3. Feature extraction ──────────────────────────────────────────────────
    print("Extracting MFCC features … (this may take a few minutes)")
    X, y, enc = build_feature_matrix(df)
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")

    n_classes = y.shape[1]

    # ── 4. Train selected models ───────────────────────────────────────────────
    for model_key in args.models:
        build_fn = MODEL_REGISTRY[model_key]
        model = build_fn(n_classes)
        train_and_evaluate(model, X, y, model_name=model_key.upper())

    print("\nAll done. Outputs saved to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
