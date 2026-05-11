"""
preprocessing.py
----------------
Handles:
  1. Computing the mean face vector.
  2. Mean-normalizing the Face Database (subtracting the mean from every image).

Why mean normalization?
  PCA finds directions of maximum variance. If we don't remove the mean,
  the first principal component will mostly capture the average brightness
  rather than the actual differences between faces. Subtracting the mean
  centres the data at the origin so PCA can focus on variance caused by
  identity differences.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def compute_mean_face(face_db: np.ndarray) -> np.ndarray:
    """
    Compute the mean face vector across all images.

    Args:
        face_db (np.ndarray): Shape (mn, p) — each column is a flattened image.

    Returns:
        mean_face (np.ndarray): Shape (mn, 1) — the average face vector.
    """
    # Mean across columns (axis=1 gives mean pixel over all images)
    mean_face = np.mean(face_db, axis=1, keepdims=True)  # (mn, 1)
    print(f"[Preprocessing] Mean face computed. Shape: {mean_face.shape}")
    return mean_face


def mean_normalize(face_db: np.ndarray, mean_face: np.ndarray) -> np.ndarray:
    """
    Subtract the mean face from every image in the database.

    Delta = Face_DB - Mean_Face   (broadcasting handles the subtraction)

    Args:
        face_db   (np.ndarray): Shape (mn, p).
        mean_face (np.ndarray): Shape (mn, 1).

    Returns:
        delta (np.ndarray): Mean-normalized matrix, same shape as face_db.
    """
    delta = face_db - mean_face  # Broadcasting: (mn,p) - (mn,1) = (mn,p)
    print(f"[Preprocessing] Mean normalization done. Delta shape: {delta.shape}")
    return delta


def visualize_mean_face(mean_face: np.ndarray, image_size: tuple,
                        output_dir: str) -> None:
    """
    Display and save the mean face image.

    Args:
        mean_face  (np.ndarray): Shape (mn, 1) or (mn,).
        image_size (tuple):      (height, width).
        output_dir (str):        Directory to save the image.
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = image_size
    mean_img = mean_face.reshape(h, w)

    # Normalise to [0, 255] for display
    mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mean_img_norm, cmap='gray')
    ax.set_title("Mean Face", fontsize=14, fontweight='bold')
    ax.axis('off')

    save_path = os.path.join(output_dir, "mean_face.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[Preprocessing] Mean face saved to: {save_path}")
