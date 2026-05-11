"""
utils.py
--------
Utility functions used across modules:
  - Project directory setup
  - Single image loading + preprocessing for inference
  - Configuration dataclass
  - Saving / loading numpy arrays
  - Summary table printing
"""

import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration for the face recognition project."""
    # Paths
    dataset_path : str = "dataset"
    output_dir   : str = "outputs"
    model_dir    : str = "models"

    # Image settings
    image_size   : tuple = (100, 100)   # (height, width)

    # PCA settings
    k_values     : List[int] = field(default_factory=lambda: [5, 10, 20, 30, 40, 50])
    default_k    : int = 30

    # Training settings
    test_size    : float = 0.4
    epochs       : int = 50
    batch_size   : int = 16
    random_state : int = 42

    # Imposter detection
    confidence_threshold : float = 0.70


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def setup_project_dirs(config: Config) -> None:
    """Create all required output directories if they don't exist."""
    dirs = [config.output_dir, config.model_dir,
            os.path.join(config.output_dir, "individual_eigenfaces")]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[Utils] Project directories ready.")


# ---------------------------------------------------------------------------
# Single-image inference helper
# ---------------------------------------------------------------------------

def load_single_image(image_path: str, image_size: tuple) -> np.ndarray:
    """
    Load, convert to grayscale, resize, and flatten a single query image.

    Args:
        image_path (str):   Path to the query image file.
        image_size (tuple): (height, width).

    Returns:
        flat_img (np.ndarray): Flattened pixel vector, shape (mn,).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError:        If the image cannot be read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (image_size[1], image_size[0]))
    flat_img = img.flatten().astype(np.float64)
    return flat_img


def preprocess_query_image(flat_img: np.ndarray, mean_face: np.ndarray,
                            feature_matrix: np.ndarray) -> np.ndarray:
    """
    Preprocess a query image for ANN prediction:
      1. Subtract mean face
      2. Project onto eigenfaces

    Args:
        flat_img       (np.ndarray): Flattened raw image (mn,).
        mean_face      (np.ndarray): Mean face vector (mn, 1).
        feature_matrix (np.ndarray): Eigenface directions (mn, k).

    Returns:
        signature (np.ndarray): Shape (1, k) — ready for model.predict().
    """
    # Mean normalise
    normalised = flat_img.reshape(-1, 1) - mean_face          # (mn, 1)

    # Project onto eigenfaces
    signature = (feature_matrix.T @ normalised).flatten()     # (k,)
    return signature.reshape(1, -1)                            # (1, k)


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_numpy(array: np.ndarray, path: str) -> None:
    np.save(path, array)
    print(f"[Utils] Saved: {path}")


def load_numpy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_accuracy_table(k_accuracy: dict) -> None:
    """Print a formatted accuracy comparison table."""
    print("\n" + "=" * 35)
    print(f"  {'k':>5}  |  {'Accuracy':>10}")
    print("-" * 35)
    for k, acc in sorted(k_accuracy.items()):
        marker = " ← best" if acc == max(k_accuracy.values()) else ""
        print(f"  {k:>5}  |  {acc * 100:>9.2f}%{marker}")
    print("=" * 35 + "\n")
