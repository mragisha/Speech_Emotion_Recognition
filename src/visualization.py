"""Plotting helpers: waveforms, mel spectrograms, MFCC heatmaps, training curves."""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import config


def plot_emotion_distribution(df: pd.DataFrame, save_dir: str = config.PLOT_SAVE) -> None:
    counts = df["Emotions"].value_counts()
    plt.figure(figsize=(12, 6), dpi=150)
    ax = sns.barplot(x=counts.index, y=counts.values, palette="mako")
    ax.bar_label(ax.containers[0], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Emotion", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Emotion Distribution", fontsize=16)
    plt.tight_layout()
    _save_or_show(os.path.join(save_dir, "emotion_distribution.png"))


def plot_waveforms(augmentations: dict, sr: int, save_dir: str = config.PLOT_SAVE) -> None:
    n = len(augmentations)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), dpi=150)
    for ax, (title, data) in zip(axes, augmentations.items()):
        ax.plot(data, color="#2E40CB")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
    plt.tight_layout()
    _save_or_show(os.path.join(save_dir, "waveforms.png"))


def plot_mel_spectrograms(augmentations: dict, sr: int, save_dir: str = config.PLOT_SAVE) -> None:
    n = len(augmentations)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
    for ax, (title, data) in zip(axes, augmentations.items()):
        spec = librosa.feature.melspectrogram(y=data, sr=sr)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        librosa.display.specshow(log_spec, x_axis="time", y_axis="mel", sr=sr,
                                 cmap="coolwarm", ax=ax)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    _save_or_show(os.path.join(save_dir, "mel_spectrograms.png"))


def plot_mfcc_heatmap(df: pd.DataFrame, extract_fn, save_dir: str = config.PLOT_SAVE) -> None:
    emotion_groups = df.groupby("Emotions")["Path"].apply(list)
    mfcc_means = {
        emotion: np.mean([extract_fn(f) for f in paths], axis=0)
        for emotion, paths in emotion_groups.items()
    }
    df_mfcc = pd.DataFrame(mfcc_means)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_mfcc, cmap="coolwarm", annot=False)
    plt.xlabel("Emotion")
    plt.ylabel("MFCC Coefficient")
    plt.title("Mean MFCC per Emotion")
    plt.tight_layout()
    _save_or_show(os.path.join(save_dir, "mfcc_heatmap.png"))


def plot_training_history(history, model_name: str, save_dir: str = config.PLOT_SAVE) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), dpi=150)

    ax1.plot(history.history["accuracy"],     label="Train",      color="blue")
    ax1.plot(history.history["val_accuracy"], label="Validation", color="orange")
    ax1.set_title(f"{model_name} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train",      color="blue")
    ax2.plot(history.history["val_loss"], label="Validation", color="orange")
    ax2.set_title(f"{model_name} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    _save_or_show(os.path.join(save_dir, f"{model_name}_history.png"))


def plot_confusion_matrix(y_true, y_pred, target_names, model_name: str,
                          save_dir: str = config.PLOT_SAVE) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    _save_or_show(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"))


def _save_or_show(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
