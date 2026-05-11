"""
data_loader.py
--------------
Handles loading face images from the dataset directory.
Each subfolder inside dataset/ is treated as one person (class).
Images are read in grayscale, resized, and flattened into column vectors.
"""

import os
import cv2
import numpy as np


def load_dataset(dataset_path: str, image_size: tuple = (100, 100)) -> tuple:
    """
    Load all face images from the dataset directory.

    Expected directory structure:
        dataset/
            person1/
                img1.jpg
                img2.jpg
            person2/
                img1.jpg
            ...

    Args:
        dataset_path (str): Path to the root dataset folder.
        image_size (tuple): Target size (height, width) for resizing images.

    Returns:
        face_db     (np.ndarray): Shape (m*n, p) — each column is a flattened image.
        labels      (np.ndarray): Integer class labels for each image.
        label_names (list):       Human-readable class names (folder names).
    """
    face_vectors = []   # Will hold flattened image vectors
    labels = []         # Integer label per image
    label_names = []    # Folder names (person names)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Each subfolder = one person
    subfolders = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if not subfolders:
        raise ValueError(f"No subfolders found in: {dataset_path}. "
                         "Each person should have their own subfolder.")

    for class_idx, folder_name in enumerate(subfolders):
        folder_path = os.path.join(dataset_path, folder_name)
        label_names.append(folder_name)

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm', '.bmp'))
        ]

        if not image_files:
            print(f"  [WARNING] No images found in: {folder_path}")
            continue

        for img_file in sorted(image_files):
            img_path = os.path.join(folder_path, img_file)

            # Read in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARNING] Could not read image: {img_path}")
                continue

            # Resize to fixed dimensions
            img = cv2.resize(img, (image_size[1], image_size[0]))

            # Flatten: (H, W) → (H*W,)
            img_vector = img.flatten().astype(np.float64)

            face_vectors.append(img_vector)
            labels.append(class_idx)

    if not face_vectors:
        raise ValueError("No images were loaded. Check your dataset structure.")

    # Stack into matrix: shape (num_pixels, num_images)
    face_db = np.array(face_vectors).T  # (mn, p)
    labels = np.array(labels)

    # --- Summary ---
    m, n = image_size
    p = face_db.shape[1]
    num_classes = len(label_names)

    print("\n" + "=" * 50)
    print("  DATASET LOADED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Total images       : {p}")
    print(f"  Image dimensions   : {m} x {n} = {m * n} pixels")
    print(f"  Number of classes  : {num_classes}")
    print(f"  Face DB shape      : {face_db.shape}  (pixels x images)")
    print(f"  Classes            : {label_names}")
    print("=" * 50 + "\n")

    return face_db, labels, label_names
