"""
ann_model.py
------------
Defines and compiles the Artificial Neural Network (ANN) classifier.

Architecture:
    Input  → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout → Dense(n_classes, Softmax)

Why this architecture?
  • Two hidden layers provide enough capacity to learn non-linear decision
    boundaries in the PCA-reduced feature space.
  • ReLU avoids the vanishing gradient problem common with sigmoid activations.
  • Dropout regularisation prevents overfitting on small face datasets.
  • Softmax output produces class probabilities (useful for confidence scoring
    and imposter detection).
  • Adam optimiser adapts learning rates per parameter — converges faster than
    vanilla SGD for this kind of problem.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_ann(input_dim: int, num_classes: int, dropout_rate: float = 0.3) -> keras.Model:
    """
    Build and compile the ANN model.

    Args:
        input_dim    (int):   Number of PCA components (k).
        num_classes  (int):   Number of face classes to recognise.
        dropout_rate (float): Dropout probability (0 = no dropout).

    Returns:
        model (keras.Model): Compiled ANN.
    """
    model = keras.Sequential([
        # Input layer (implicitly created by specifying input_shape)
        layers.Input(shape=(input_dim,), name="pca_features"),

        # Hidden layer 1
        layers.Dense(128, activation='relu', name="hidden_1",
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),

        # Hidden layer 2
        layers.Dense(64, activation='relu', name="hidden_2",
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(dropout_rate, name="dropout_2"),

        # Output layer
        layers.Dense(num_classes, activation='softmax', name="output"),
    ], name="FaceRecognition_ANN")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def get_callbacks(model_dir: str, patience: int = 10) -> list:
    """
    Return a list of Keras training callbacks.

    Includes:
      - ModelCheckpoint: saves the best model weights.
      - EarlyStopping:   stops training if val_loss doesn't improve.
      - ReduceLROnPlateau: lowers learning rate when plateauing.

    Args:
        model_dir (str): Directory to save checkpoint weights.
        patience  (int): Number of epochs to wait before early stopping.

    Returns:
        List of callback objects.
    """
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, "best_ann_weights.weights.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    return callbacks
