"""
trainer.py
----------
Handles:
  1. Train/test split of face signatures.
  2. Training the ANN with callbacks.
  3. Plotting and saving training history (accuracy + loss curves).
  4. Saving the final trained model.
  5. Saving PCA artifacts and train config for use by test.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras

from src.ann_model import build_ann, get_callbacks


def prepare_data(signatures: np.ndarray, labels: np.ndarray,
                 test_size: float = 0.4, random_state: int = 42) -> tuple:
    """
    Encode labels and split into training and test sets.

    Args:
        signatures   (np.ndarray): Shape (p, k) — PCA-projected face vectors.
        labels       (np.ndarray): Integer class labels (p,).
        test_size    (float):      Fraction of data for testing (default 40%).
        random_state (int):        Reproducibility seed.

    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        signatures, encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels
    )

    print(f"[Trainer] Data split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, le


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                num_classes: int, model_dir: str,
                epochs: int = 50, batch_size: int = 16) -> tuple:
    """
    Build, train and return the ANN model along with its training history.

    Args:
        X_train, y_train: Training data.
        X_test,  y_test:  Validation data.
        num_classes (int): Number of face classes.
        model_dir   (str): Directory to save model.
        epochs      (int): Maximum training epochs.
        batch_size  (int): Mini-batch size.

    Returns:
        model   (keras.Model):  Trained ANN.
        history (History):      Keras training history object.
    """
    input_dim = X_train.shape[1]
    model = build_ann(input_dim, num_classes)
    callbacks = get_callbacks(model_dir)

    print(f"\n[Trainer] Training ANN  |  epochs={epochs}  batch={batch_size}")
    print(f"          Input dim = {input_dim}  |  Classes = {num_classes}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save the full model (architecture + weights)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "face_recognition_ann.h5")
    model.save(model_path)
    print(f"\n[Trainer] Model saved to: {model_path}")

    return model, history


def save_artifacts(feature_matrix: np.ndarray, label_names: list,
                   model_dir: str, k: int,
                   image_size: tuple = (100, 100),
                   confidence_threshold: float = 0.70) -> None:
    """
    Save PCA feature matrix, label names, and training configuration
    to the model directory so that test.py can load them without retraining.

    Args:
        feature_matrix        (np.ndarray): Shape (mn, k) top-k eigenfaces.
        label_names           (list):       Human-readable class names.
        model_dir             (str):        Directory to save artifacts.
        k                     (int):        Number of principal components used.
        image_size            (tuple):      (height, width) used during training.
        confidence_threshold  (float):      Threshold used for imposter detection.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Save feature matrix (top-k eigenfaces)
    fm_path = os.path.join(model_dir, "feature_matrix.npy")
    np.save(fm_path, feature_matrix)
    print(f"[Trainer] Feature matrix saved to: {fm_path}")

    # Save label names
    ln_path = os.path.join(model_dir, "label_names.npy")
    np.save(ln_path, np.array(label_names))
    print(f"[Trainer] Label names saved to: {ln_path}")

    # Save training config as JSON
    config = {
        "k": k,
        "image_size": list(image_size),
        "confidence_threshold": confidence_threshold,
        "num_classes": len(label_names),
        "label_names": label_names
    }
    cfg_path = os.path.join(model_dir, "train_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Trainer] Train config saved to: {cfg_path}")


def plot_training_history(history, output_dir: str, k: int = None) -> None:
    """
    Plot and save training accuracy and loss curves.

    Args:
        history    (History): Keras training history.
        output_dir (str):     Directory to save plots.
        k          (int):     k value used (for labelling the plot filename).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    suffix = f"_k{k}" if k else ""

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title(f"Training Accuracy{' (k=' + str(k) + ')' if k else ''}",
                  fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title(f"Training Loss{' (k=' + str(k) + ')' if k else ''}",
                  fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"training_history{suffix}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[Trainer] Training history plot saved to: {save_path}")
