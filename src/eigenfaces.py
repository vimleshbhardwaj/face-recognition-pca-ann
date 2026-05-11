"""
eigenfaces.py
-------------
Generates, visualises, and saves eigenfaces.

Eigenfaces are the principal components (eigenvectors) of the face dataset
reshaped back into image form. They look like ghostly face-like images and
capture the most important modes of variation across the dataset.

The first eigenface captures the direction of greatest variance (e.g. overall
lighting differences), and subsequent eigenfaces capture progressively subtler
variations (expression, pose, identity features, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def normalize_for_display(vector: np.ndarray) -> np.ndarray:
    """
    Normalise a vector to [0, 1] range for display purposes.
    """
    v_min, v_max = vector.min(), vector.max()
    if v_max - v_min < 1e-8:
        return np.zeros_like(vector)
    return (vector - v_min) / (v_max - v_min)


def visualize_eigenfaces(feature_matrix: np.ndarray, image_size: tuple,
                          output_dir: str, num_to_show: int = 10) -> None:
    """
    Display and save the first `num_to_show` eigenfaces.

    Args:
        feature_matrix (np.ndarray): Shape (mn, k) — columns are eigenface directions.
        image_size     (tuple):      (height, width).
        output_dir     (str):        Directory to save images.
        num_to_show    (int):        How many eigenfaces to visualise.
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = image_size
    k = feature_matrix.shape[1]
    n_display = min(num_to_show, k)

    # Grid layout
    cols = 5
    rows = (n_display + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(min(cols * 2.2, 11), min(rows * 2.2, 5)))
    fig.suptitle(f"First {n_display} Eigenfaces", fontsize=14, fontweight='bold')

    axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_display:
            eigenface = feature_matrix[:, i]
            eigenface_img = normalize_for_display(eigenface).reshape(h, w)
            ax.imshow(eigenface_img, cmap='gray')
            ax.set_title(f"EF #{i + 1}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "eigenfaces_grid.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[Eigenfaces] Grid saved to: {save_path}")

    # Also save each eigenface individually
    individual_dir = os.path.join(output_dir, "individual_eigenfaces")
    os.makedirs(individual_dir, exist_ok=True)
    for i in range(n_display):
        ef = normalize_for_display(feature_matrix[:, i]).reshape(h, w)
        save_individual = os.path.join(individual_dir, f"eigenface_{i + 1:02d}.png")
        plt.imsave(save_individual, ef, cmap='gray')

    print(f"[Eigenfaces] Individual eigenfaces saved to: {individual_dir}")
