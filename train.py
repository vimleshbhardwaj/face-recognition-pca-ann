"""
train.py
========
Standalone script to train the PCA + ANN Face Recognition System.
It runs the pipeline, trains the model, and saves all necessary artifacts
to the `models/` directory so they can be used by `test.py` later.

Usage:
    python train.py
    python train.py --dataset dataset --k 30 --epochs 50
"""

import argparse
import os
import sys

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import Config, setup_project_dirs, save_numpy
from src.data_loader import load_dataset
from src.preprocessing import compute_mean_face, mean_normalize, visualize_mean_face
from src.pca_module import (compute_surrogate_covariance, compute_eigenvectors,
                             recover_eigenfaces_directions, select_top_k,
                             project_faces, plot_eigenvalue_distribution)
from src.eigenfaces import visualize_eigenfaces
from src.trainer import prepare_data, train_model, plot_training_history, save_artifacts

def parse_args():
    parser = argparse.ArgumentParser(description="Train PCA + ANN Model")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset folder")
    parser.add_argument("--k", type=int, default=30, help="Number of principal components")
    parser.add_argument("--epochs", type=int, default=50, help="ANN training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="ANN training batch size")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        dataset_path=args.dataset,
        default_k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        confidence_threshold=args.threshold,
    )

    print("\n" + "=" * 60)
    print("  PCA + ANN TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Dataset  : {cfg.dataset_path}")
    print(f"  k        : {cfg.default_k}")
    print(f"  Epochs   : {cfg.epochs}")
    print("=" * 60 + "\n")

    setup_project_dirs(cfg)

    print("\n─── STEP 1: Loading Dataset ───")
    face_db, labels, label_names = load_dataset(cfg.dataset_path, cfg.image_size)

    print("\n─── STEP 2: Computing Mean Face ───")
    mean_face = compute_mean_face(face_db)
    visualize_mean_face(mean_face, cfg.image_size, cfg.output_dir)

    print("\n─── STEP 3: Mean Normalization ───")
    delta = mean_normalize(face_db, mean_face)

    print("\n─── STEP 4: Surrogate Covariance Matrix ───")
    C = compute_surrogate_covariance(delta)

    print("\n─── STEP 5: Eigen-decomposition ───")
    eigenvalues, eigenvectors = compute_eigenvectors(C)
    plot_eigenvalue_distribution(eigenvalues, cfg.output_dir)

    eigenface_dirs = recover_eigenfaces_directions(delta, eigenvectors)

    save_numpy(eigenface_dirs, os.path.join(cfg.model_dir, "eigenface_dirs.npy"))
    save_numpy(mean_face,      os.path.join(cfg.model_dir, "mean_face.npy"))
    save_numpy(labels,         os.path.join(cfg.model_dir, "labels.npy"))

    print(f"\n─── STEP 6: Selecting top-k={cfg.default_k} features ───")
    feature_matrix = select_top_k(eigenface_dirs, cfg.default_k)

    print("\n─── STEP 7: Generating Eigenfaces ───")
    visualize_eigenfaces(feature_matrix, cfg.image_size, cfg.output_dir, num_to_show=10)

    print("\n─── STEP 8: Generating Face Signatures ───")
    signatures = project_faces(delta, feature_matrix)

    print("\n─── STEP 9: Training ANN ───")
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(
        signatures, labels, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model, history = train_model(
        X_train, y_train, X_test, y_test,
        num_classes=len(label_names),
        model_dir=cfg.model_dir,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size
    )

    plot_training_history(history, cfg.output_dir, k=cfg.default_k)

    print("\n─── Save Artifacts for test.py ───")
    save_artifacts(feature_matrix, label_names,
                   model_dir=cfg.model_dir,
                   k=cfg.default_k,
                   image_size=cfg.image_size,
                   confidence_threshold=cfg.confidence_threshold)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Model and artifacts saved to: {cfg.model_dir}/")
    print("  Run 'python test.py' to evaluate.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
