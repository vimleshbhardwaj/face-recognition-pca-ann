"""
main.py
=======
Face Recognition System — Main Pipeline
========================================

Orchestrates the full pipeline:
  Step 1  → Load Dataset
  Step 2  → Compute Mean Face
  Step 3  → Mean Normalization
  Step 4  → Surrogate Covariance Matrix
  Step 5  → Eigen-decomposition
  Step 6  → Feature Vector Selection
  Step 7  → Generate Eigenfaces
  Step 8  → Generate Face Signatures
  Step 9  → Train ANN
  Step 10 → Test on a single image
  Step 11 → Accuracy vs k Analysis
  Step 12 → Imposter Detection
  Step 13 → Evaluation Metrics
  Step 14 → All Visualizations saved

Usage:
    python main.py
    python main.py --dataset dataset --k 30 --epochs 50
"""

import argparse
import os
import sys
import numpy as np

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import (Config, setup_project_dirs, load_single_image,
                        preprocess_query_image, save_numpy, print_accuracy_table)
from src.data_loader import load_dataset
from src.preprocessing import compute_mean_face, mean_normalize, visualize_mean_face
from src.pca_module import (compute_surrogate_covariance, compute_eigenvectors,
                             recover_eigenfaces_directions, select_top_k,
                             project_faces, plot_eigenvalue_distribution)
from src.eigenfaces import visualize_eigenfaces
from src.trainer import prepare_data, train_model, plot_training_history, save_artifacts
from src.evaluator import (evaluate_model, run_imposter_demo, accuracy_vs_k,
                            predict_with_imposter_detection)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA + ANN Face Recognition System"
    )
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Path to dataset folder (default: dataset/)")
    parser.add_argument("--k", type=int, default=30,
                        help="Number of principal components (default: 30)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="ANN training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="ANN training batch size (default: 16)")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Confidence threshold for imposter detection (default: 0.60)")
    parser.add_argument("--skip_k_analysis", action="store_true",
                        help="Skip the accuracy vs k sweep (saves time)")
    parser.add_argument("--test_image", type=str, default=None,
                        help="Path to a single test image for demo prediction")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Build config from CLI args
    cfg = Config(
        dataset_path=args.dataset,
        default_k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        confidence_threshold=args.threshold,
    )

    print("\n" + "=" * 60)
    print("  PCA + ANN FACE RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"  Dataset  : {cfg.dataset_path}")
    print(f"  k        : {cfg.default_k}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Batch    : {cfg.batch_size}")
    print(f"  Threshold: {cfg.confidence_threshold}")
    print("=" * 60 + "\n")

    # ── Setup ──────────────────────────────────────────────────────────────
    setup_project_dirs(cfg)

    # ── STEP 1: Load Dataset ───────────────────────────────────────────────
    print("\n─── STEP 1: Loading Dataset ───")
    face_db, labels, label_names = load_dataset(cfg.dataset_path, cfg.image_size)

    # ── STEP 2: Mean Face ──────────────────────────────────────────────────
    print("\n─── STEP 2: Computing Mean Face ───")
    mean_face = compute_mean_face(face_db)
    visualize_mean_face(mean_face, cfg.image_size, cfg.output_dir)

    # ── STEP 3: Mean Normalization ─────────────────────────────────────────
    print("\n─── STEP 3: Mean Normalization ───")
    delta = mean_normalize(face_db, mean_face)

    # ── STEP 4: Surrogate Covariance Matrix ───────────────────────────────
    print("\n─── STEP 4: Surrogate Covariance Matrix ───")
    C = compute_surrogate_covariance(delta)

    # ── STEP 5: Eigen-decomposition ────────────────────────────────────────
    print("\n─── STEP 5: Eigen-decomposition ───")
    eigenvalues, eigenvectors = compute_eigenvectors(C)
    plot_eigenvalue_distribution(eigenvalues, cfg.output_dir)

    # ── STEP 5b: Recover true eigenface directions ─────────────────────────
    eigenface_dirs = recover_eigenfaces_directions(delta, eigenvectors)

    # Save for later use (accuracy_vs_k or reloading)
    save_numpy(eigenface_dirs, os.path.join(cfg.model_dir, "eigenface_dirs.npy"))
    save_numpy(mean_face,      os.path.join(cfg.model_dir, "mean_face.npy"))
    save_numpy(labels,         os.path.join(cfg.model_dir, "labels.npy"))

    # ── STEP 6: Feature Vector Selection ──────────────────────────────────
    print(f"\n─── STEP 6: Selecting top-k={cfg.default_k} features ───")
    feature_matrix = select_top_k(eigenface_dirs, cfg.default_k)

    # ── STEP 7: Visualise Eigenfaces ──────────────────────────────────────
    print("\n─── STEP 7: Generating Eigenfaces ───")
    visualize_eigenfaces(feature_matrix, cfg.image_size, cfg.output_dir, num_to_show=10)

    # ── STEP 8: Generate Face Signatures ──────────────────────────────────
    print("\n─── STEP 8: Generating Face Signatures ───")
    signatures = project_faces(delta, feature_matrix)

    # ── STEP 9: Train ANN ─────────────────────────────────────────────────
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

    # ── Save Artifacts for test.py ────────────────────────────────────────
    save_artifacts(feature_matrix, label_names,
                   model_dir=cfg.model_dir,
                   k=cfg.default_k,
                   image_size=cfg.image_size,
                   confidence_threshold=cfg.confidence_threshold)

    # ── STEP 10: Single Image Test ─────────────────────────────────────────
    if args.test_image:
        print(f"\n─── STEP 10: Testing Single Image ({args.test_image}) ───")
        try:
            flat_img = load_single_image(args.test_image, cfg.image_size)
            sig = preprocess_query_image(flat_img, mean_face, feature_matrix)
            predict_with_imposter_detection(model, sig, label_names,
                                            threshold=cfg.confidence_threshold)
        except Exception as e:
            print(f"[WARNING] Could not test image: {e}")
    else:
        print("\n─── STEP 10: Single image test skipped (no --test_image provided) ───")

    # ── STEP 11: Accuracy vs k ─────────────────────────────────────────────
    if not args.skip_k_analysis:
        print("\n─── STEP 11: Accuracy vs k Analysis ───")
        k_accuracy = accuracy_vs_k(
            face_db=face_db,
            labels=labels,
            mean_face=mean_face,
            eigenface_dirs=eigenface_dirs,
            label_names=label_names,
            k_values=cfg.k_values,
            output_dir=cfg.output_dir,
            model_dir=cfg.model_dir,
            epochs=30,           # Fewer epochs for speed during sweep
            batch_size=cfg.batch_size
        )
        print_accuracy_table(k_accuracy)
    else:
        print("\n─── STEP 11: Skipped (--skip_k_analysis) ───")

    # ── STEP 12: Imposter Detection Demo ──────────────────────────────────
    print("\n─── STEP 12: Imposter Detection ───")
    run_imposter_demo(model, X_test, y_test, label_names,
                      cfg.output_dir,
                      mean_face=mean_face,
                      feature_matrix=feature_matrix,
                      imposters_dir="imposters",
                      threshold=cfg.confidence_threshold)

    # ── STEP 13: Evaluation Metrics ───────────────────────────────────────
    print("\n─── STEP 13: Evaluation Metrics ───")
    metrics = evaluate_model(model, X_test, y_test, label_names, cfg.output_dir)

    # ── Final Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Final Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Final Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"  Final F1-Score  : {metrics['f1'] * 100:.2f}%")
    print(f"\n  Outputs saved to : {cfg.output_dir}/")
    print(f"  Model saved to   : {cfg.model_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
