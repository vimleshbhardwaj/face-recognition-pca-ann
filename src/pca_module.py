"""
pca_module.py
-------------
Manual PCA implementation using:
  1. Surrogate (compact) covariance matrix  C = Delta^T * Delta
  2. NumPy eigen-decomposition
  3. Projection of data onto selected eigenvectors

Why surrogate covariance?
  The true covariance matrix would be (mn x mn) — for 100x100 images that
  is 10,000 x 10,000 = 100 million elements. This is prohibitively large.
  Instead we compute C = Delta^T * Delta which is only (p x p) where p is
  the number of images (typically much smaller than mn). The eigenvectors
  of the true covariance can be recovered from those of C by the relation:
      u_i = Delta * v_i   (then normalised)
  This is the mathematical trick introduced by Turk & Pentland (1991).
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Step 1: Surrogate Covariance Matrix
# ---------------------------------------------------------------------------

def compute_surrogate_covariance(delta: np.ndarray) -> np.ndarray:
    """
    Compute the surrogate (compact) covariance matrix.

    C = Delta^T x Delta    shape: (p, p)

    Args:
        delta (np.ndarray): Mean-normalised face matrix, shape (mn, p).

    Returns:
        C (np.ndarray): Surrogate covariance matrix, shape (p, p).
    """
    C = delta.T @ delta  # (p, mn) @ (mn, p) = (p, p)

    print("\n[PCA] Surrogate covariance matrix computed.")
    print(f"      Delta shape : {delta.shape}  (pixels x images)")
    print(f"      C shape     : {C.shape}  (images x images)  ← much smaller!")
    print("      (Avoids the huge mn×mn true covariance matrix)\n")

    return C


# ---------------------------------------------------------------------------
# Step 2: Eigen-decomposition
# ---------------------------------------------------------------------------

def compute_eigenvectors(C: np.ndarray) -> tuple:
    """
    Compute and sort eigenvalues + eigenvectors of surrogate covariance.

    Args:
        C (np.ndarray): Surrogate covariance matrix (p, p).

    Returns:
        eigenvalues  (np.ndarray): Sorted descending, shape (p,).
        eigenvectors (np.ndarray): Corresponding column vectors, shape (p, p).
    """
    # numpy.linalg.eig returns complex values for non-symmetric matrices;
    # we take the real part (imaginary parts are numerical noise).
    eigenvalues, eigenvectors = np.linalg.eig(C)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort in descending order of eigenvalue magnitude
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    print(f"[PCA] Eigen-decomposition complete.")
    print(f"      Number of eigenvalues : {len(eigenvalues)}")
    print(f"      Top-5 eigenvalues     : {eigenvalues[:5]}\n")

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Step 3: Recover true eigenvectors (eigenfaces directions)
# ---------------------------------------------------------------------------

def recover_eigenfaces_directions(delta: np.ndarray,
                                   eigenvectors: np.ndarray) -> np.ndarray:
    """
    Recover the true high-dimensional eigenvectors (eigenface directions)
    from the surrogate eigenvectors.

    u_i = Delta * v_i   then normalised to unit length.

    Args:
        delta       (np.ndarray): (mn, p)
        eigenvectors(np.ndarray): Surrogate eigenvectors (p, p).

    Returns:
        eigenface_dirs (np.ndarray): Shape (mn, p) — columns are eigenface dirs.
    """
    # Project: (mn, p) @ (p, p) = (mn, p)
    eigenface_dirs = delta @ eigenvectors

    # Normalise each column to unit length
    norms = np.linalg.norm(eigenface_dirs, axis=0, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    eigenface_dirs = eigenface_dirs / norms

    print(f"[PCA] True eigenface directions recovered. Shape: {eigenface_dirs.shape}")
    return eigenface_dirs


# ---------------------------------------------------------------------------
# Step 4: Select top-k feature vectors
# ---------------------------------------------------------------------------

def select_top_k(eigenface_dirs: np.ndarray, k: int) -> np.ndarray:
    """
    Select the top-k eigenface direction vectors (feature vector matrix).

    Args:
        eigenface_dirs (np.ndarray): All eigenface directions (mn, p).
        k (int): Number of principal components to keep.

    Returns:
        feature_matrix (np.ndarray): Shape (mn, k).
    """
    feature_matrix = eigenface_dirs[:, :k]
    print(f"[PCA] Selected top-k={k} eigenvectors. Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix


# ---------------------------------------------------------------------------
# Step 5: Project images → face signatures
# ---------------------------------------------------------------------------

def project_faces(delta: np.ndarray, feature_matrix: np.ndarray) -> np.ndarray:
    """
    Project mean-normalised faces onto the eigenface subspace to get
    low-dimensional 'face signatures' (weight vectors).

    Signatures = Feature_Matrix^T x Delta
               = (k, mn) @ (mn, p) = (k, p)

    Each column is the k-dimensional representation of one face.

    Args:
        delta          (np.ndarray): (mn, p)
        feature_matrix (np.ndarray): (mn, k)

    Returns:
        signatures (np.ndarray): Shape (p, k)  — one row per image.
    """
    signatures = (feature_matrix.T @ delta).T  # (p, k)
    print(f"[PCA] Face signatures generated. Shape: {signatures.shape}  (images x k)")
    return signatures


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def plot_eigenvalue_distribution(eigenvalues: np.ndarray, output_dir: str) -> None:
    """
    Plot and save the eigenvalue distribution (scree plot).
    Helps decide how many principal components to retain.
    """
    os.makedirs(output_dir, exist_ok=True)

    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Individual eigenvalues
    ax1.plot(eigenvalues[:50], 'bo-', markersize=4, linewidth=1.5)
    ax1.set_title("Eigenvalue Distribution (Top 50)", fontweight='bold')
    ax1.set_xlabel("Component Index")
    ax1.set_ylabel("Eigenvalue")
    ax1.grid(True, alpha=0.3)

    # Cumulative explained variance
    ax2.plot(cumulative_variance[:50], 'r-', linewidth=2)
    ax2.axhline(y=95, color='gray', linestyle='--', label='95% variance')
    ax2.set_title("Cumulative Explained Variance", fontweight='bold')
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Variance Explained (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "eigenvalue_distribution.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[PCA] Eigenvalue distribution plot saved to: {save_path}")
