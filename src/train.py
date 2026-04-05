"""Training and evaluation pipeline for a single model."""

import os
import numpy as np
from sklearn.metrics import classification_report

import config
from src.visualization import plot_training_history, plot_confusion_matrix


def train_and_evaluate(model, X, y, model_name: str, target_names=config.TARGET_NAMES):
    """
    Train `model` on (X, y) with the settings in config, then print
    the classification report and save plots.

    Returns
    -------
    history : keras History object
    """
    os.makedirs(config.MODEL_SAVE, exist_ok=True)
    os.makedirs(config.PLOT_SAVE, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")
    model.summary()

    history = model.fit(
        X, y,
        validation_split=config.VALIDATION_SPLIT,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        verbose=1,
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_training_history(history, model_name)

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\n[{model_name}] Overall  Loss: {loss:.6f}  Accuracy: {acc:.6f}")

    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y, axis=1)

    plot_confusion_matrix(y_true, y_pred, target_names, model_name)

    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ── Save model ─────────────────────────────────────────────────────────────
    save_path = os.path.join(config.MODEL_SAVE, f"{model_name}.keras")
    model.save(save_path)
    print(f"Model saved → {save_path}")

    return history
