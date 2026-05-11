"""
test.py
=======
Standalone script to load the trained PCA + ANN model and run predictions.
Does NOT retrain the model.

Modes:
  1. Full Evaluation: Run test.py without args (evaluates test set split + imposter demo)
  2. Single Image:    Run test.py --image path/to/image.jpg
  3. Folder:          Run test.py --folder path/to/folder/
"""

import argparse
import os
import sys
import json
import numpy as np

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from tensorflow.keras.models import load_model

from src.utils import load_numpy, load_single_image, preprocess_query_image
from src.evaluator import (evaluate_model, run_imposter_demo, predict_with_imposter_detection)
from src.trainer import prepare_data
from src.pca_module import project_faces

def parse_args():
    parser = argparse.ArgumentParser(description="Test PCA + ANN Model")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image for prediction")
    parser.add_argument("--folder", type=str, default=None, help="Path to a folder of images for prediction")
    return parser.parse_args()

def main():
    args = parse_args()

    model_dir = "models"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  PCA + ANN TESTING / INFERENCE")
    print("=" * 60)

    # 1. Load config
    config_path = os.path.join(model_dir, "train_config.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found at {config_path}.")
        print("Please run 'python train.py' first to generate model artifacts.")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    k = cfg["k"]
    image_size = tuple(cfg["image_size"])
    threshold = cfg["confidence_threshold"]
    label_names = cfg["label_names"]

    print(f"  Loaded Config → k: {k}, image_size: {image_size}, threshold: {threshold}")

    # 2. Load PCA artifacts
    try:
        feature_matrix = load_numpy(os.path.join(model_dir, "feature_matrix.npy"))
        mean_face = load_numpy(os.path.join(model_dir, "mean_face.npy"))
    except FileNotFoundError as e:
        print(f"[ERROR] Missing artifact: {e}")
        print("Please run 'python train.py' first.")
        sys.exit(1)

    # 3. Load Model
    model_path = os.path.join(model_dir, "face_recognition_ann.h5")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}.")
        print("Please run 'python train.py' first.")
        sys.exit(1)

    print("  Loading trained ANN model...\n")
    model = load_model(model_path)

    # --- Mode 2: Single Image Inference ---
    if args.image:
        print(f"─── Evaluating Single Image: {args.image} ───")
        try:
            flat_img = load_single_image(args.image, image_size)
            sig = preprocess_query_image(flat_img, mean_face, feature_matrix)
            predict_with_imposter_detection(model, sig, label_names, threshold=threshold)
        except Exception as e:
            print(f"[ERROR] {e}")
        return

    # --- Mode 3: Folder Inference ---
    if args.folder:
        print(f"─── Evaluating Folder: {args.folder} ───")
        if not os.path.isdir(args.folder):
            print(f"[ERROR] Folder not found: {args.folder}")
            return
            
        img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.pgm')
        files = [f for f in os.listdir(args.folder) if f.lower().endswith(img_extensions)]
        
        if not files:
            print("No images found in folder.")
            return
            
        for f in files:
            fpath = os.path.join(args.folder, f)
            print(f"\nImage: {f}")
            try:
                flat_img = load_single_image(fpath, image_size)
                sig = preprocess_query_image(flat_img, mean_face, feature_matrix)
                predict_with_imposter_detection(model, sig, label_names, threshold=threshold)
            except Exception as e:
                print(f"[WARN] Failed to process {f}: {e}")
        return

    # --- Mode 1: Full Evaluation (Test Split) ---
    print("─── Full Evaluation on Test Split ───")
    
    try:
        labels = load_numpy(os.path.join(model_dir, "labels.npy"))
    except FileNotFoundError:
        print("[ERROR] labels.npy missing. Required for full evaluation.")
        sys.exit(1)
        
    from src.data_loader import load_dataset
    from src.preprocessing import mean_normalize
    try:
        print("  Loading dataset to recreate test split...")
        face_db, _, _ = load_dataset("dataset", image_size)
        delta = mean_normalize(face_db, mean_face)
        signatures = project_faces(delta, feature_matrix)
        
        # Split using the same random state
        _, X_test, _, y_test, _ = prepare_data(
            signatures, labels, test_size=0.4, random_state=42
        )
        
        evaluate_model(model, X_test, y_test, label_names, output_dir)
        
        run_imposter_demo(model, X_test, y_test, label_names,
                          output_dir,
                          mean_face=mean_face,
                          feature_matrix=feature_matrix,
                          imposters_dir="imposters",
                          threshold=threshold)
                          
        print(f"\n[Info] Evaluation results saved to {output_dir}/")
        
    except Exception as e:
        print(f"[ERROR] Could not complete full evaluation: {e}")

if __name__ == "__main__":
    main()
