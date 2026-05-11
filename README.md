# Face Recognition System — PCA (Eigenfaces) + ANN

A complete, modular face recognition pipeline built from scratch using
**Principal Component Analysis (PCA / Eigenfaces)** for dimensionality reduction
and an **Artificial Neural Network (ANN)** for classification.

Tested and working on **Python 3.13 / Windows 11** with TensorFlow 2.13+.

---

## Project Overview

This system implements the classic **Eigenfaces** approach introduced by
Turk & Pentland (1991), extended with a modern deep-learning classifier.

Instead of comparing raw pixel values, it:

1. Learns a compact **"face space"** using PCA (keeping only the most expressive directions).
2. Represents each face as a small vector of weights in that space (**face signature**).
3. Trains a neural network to map signatures → identities.
4. Rejects unknown persons using a **confidence threshold** (imposter detection).

---

## Mathematical Background

### PCA — Why and How

Given a face database matrix **Face_DB** of shape *(mn × p)*
(each of the *p* images is a flattened *m×n* pixel vector):

| Step | Formula | Description |
|------|---------|-------------|
| 1 | `M = mean(Face_DB, axis=1)` | Mean face |
| 2 | `Δ = Face_DB − M` | Mean normalization |
| 3 | `C = ΔᵀΔ` *(p×p)* | Surrogate covariance (avoids huge mn×mn matrix) |
| 4 | `C vᵢ = λᵢ vᵢ` | Eigen-decomposition |
| 5 | `uᵢ = Δ vᵢ` (normalised) | Recover true eigenface directions |
| 6 | `wᵢ = Uᵏᵀ Δᵢ` | Face signature (k-dimensional projection) |

### ANN Architecture

```
Input (k)  →  Dense(128, ReLU) → BatchNorm → Dropout(0.3)
           →  Dense(64,  ReLU) → BatchNorm → Dropout(0.3)
           →  Dense(n_classes, Softmax)
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Cross-Entropy |
| Early Stopping | Monitors `val_loss` (patience=10) |
| LR Reduction | ReduceLROnPlateau (factor=0.5, patience=5) |
| Model Checkpoint | Saves best `val_accuracy` weights |

---

## Project Structure

```
face_recognition_project/
│
├── dataset/                  ← Face images (one subfolder per person)
│   ├── Aamir/
│   ├── Ajay/
│   └── ...
│
├── imposters/                ← Images of unknown people to test rejection
├── outputs/                  ← All generated plots and visualizations
├── models/                   ← Saved ANN weights, PCA artifacts, and config
│
├── src/
│   ├── data_loader.py        ← Load images, build Face_DB matrix
│   ├── preprocessing.py      ← Mean face, mean normalization
│   ├── pca_module.py         ← Manual PCA (surrogate cov, eigen-decomp, projection)
│   ├── eigenfaces.py         ← Eigenface visualization and saving
│   ├── ann_model.py          ← Keras ANN architecture and callbacks
│   ├── trainer.py            ← Train/test split, training loop, history plots, artifact saving
│   ├── evaluator.py          ← Metrics, confusion matrix, imposter detection, k-sweep
│   └── utils.py              ← Config, helpers, single-image inference
│
├── train.py                  ← Standalone Training Script (Run this first)
├── test.py                   ← Standalone Testing Script (Run inference without retraining)
├── main.py                   ← Legacy full pipeline entry point (does both train+test at once)
├── requirements.txt
├── report.md
└── README.md
```

---

## Dataset

### Structure
The system expects a nested folder structure where each subfolder corresponds to a different person (class):
```
dataset/
├── Aamir/       ← 50 images
├── Ajay/        ← 50 images
├── Akshay/      ← 50 images
├── Alia/        ← 50 images
├── Amitabh/     ← 50 images
├── Deepika/     ← 50 images
├── Disha/       ← 50 images
├── Farhan/      ← 50 images
└── Ileana/      ← 50 images
```

> **Total: 450 images across 9 Bollywood celebrity classes (100×100 pixels each)**

---

## Installation

```powershell
# 1. Navigate to the project folder
cd "C:\path\to\face_recognition_project"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it (Windows)
.\venv\Scripts\activate

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow (~350 MB) will take a few minutes to download.
> Python **3.9 – 3.13** is supported. GPU is not used on native Windows (CPU only).

---

## How to Run

This project separates training and testing into two clear steps:

### Step 1: Train the Model

Run the training script to load the dataset, compute PCA, train the neural network, and save all artifacts (models, matrices, config) into the `models/` directory.

```powershell
python train.py
```

**Custom parameters:**
```powershell
python train.py --k 40 --epochs 60 --threshold 0.65
```

### Step 2: Test / Inference

Once the model is trained and saved, use `test.py` to evaluate it or run predictions **instantly** without retraining.

**Full Evaluation (Test split + Imposter Demo):**
```powershell
python test.py
```

**Test a Single Face Image:**
```powershell
python test.py --image dataset/Aamir/img1.jpg
```

**Test an Entire Folder (e.g., Imposters):**
```powershell
python test.py --folder imposters/
```

> *(Note: The older `python main.py` script is still available if you wish to run the entire pipeline—training and testing with K-sweeps—in a single command).*

---

## Pipeline Steps (Training)

During `train.py`, the pipeline runs automatically:

| Step | Description |
|------|-------------|
| 1 | Load Dataset — build Face_DB matrix (10000 × 450) |
| 2 | Compute Mean Face |
| 3 | Mean Normalization — Δ = Face_DB − Mean |
| 4 | Surrogate Covariance Matrix — C = ΔᵀΔ (450×450) |
| 5 | Eigen-decomposition — sort eigenvalues descending |
| 6 | Select top-k feature vectors |
| 7 | Visualize first 10 Eigenfaces |
| 8 | Generate Face Signatures (project onto k-space) |
| 9 | Train ANN (with early stopping & checkpointing) |
| 10 | Save all artifacts (`feature_matrix.npy`, `train_config.json`, etc.) |

---

## Generated Outputs

All images are saved to the `outputs/` folder:

| File | Description |
|------|-------------|
| `mean_face.png` | Average face across the entire dataset |
| `eigenvalue_distribution.png` | Scree plot + cumulative variance explained |
| `eigenfaces_grid.png` | First 10 eigenfaces in a grid |
| `individual_eigenfaces/` | Each eigenface saved individually |
| `training_history_k30.png` | Training vs validation accuracy and loss curves |
| `accuracy_vs_k.png` | Classification accuracy sweep (generated via `main.py`) |
| `confusion_matrix.png` | Predicted vs true labels heatmap |
| `imposter_detection_demo.png` | Genuine recognition vs imposter rejection demo |

Trained model files saved to `models/`:

| File | Description |
|------|-------------|
| `face_recognition_ann.h5` | Full trained ANN model |
| `best_ann_weights.weights.h5` | Best checkpoint weights |
| `eigenface_dirs.npy` | Eigenface direction matrix |
| `feature_matrix.npy` | Top-k eigenfaces matrix (loaded by test.py) |
| `mean_face.npy` | Saved mean face vector |
| `labels.npy` | Encoded label array |
| `label_names.npy` | Dataset class names |
| `train_config.json` | Configuration values (k, image_size, threshold) |

---

## Results

### Accuracy vs k (on this dataset, 40% test split)

| k | Accuracy |
|---|---------|
| 5  | 39.44% |
| 10 | 50.56% |
| 20 | 50.00% |
| 30 | 61.11% |
| 40 | 55.00% |
| **50** | **63.33% ← best** |

### Evaluation Metrics (k=30, 50 epochs)

| Metric | Value |
|--------|-------|
| Accuracy | ~60% |
| Precision | ~60% |
| Recall | ~60% |
| F1-Score | ~60% |

> Results vary slightly between runs due to random train/test splits and ANN weight initialization.
> Accuracy improves with more images per person and more training epochs.

---

## Known Notes

- `Iris` and `faces` subfolders in the dataset contain no images and are skipped automatically.
- GPU acceleration is **not available** on native Windows with TensorFlow ≥ 2.11. For GPU support, use WSL2 or the TensorFlow-DirectML plugin.
- The `WARNING: oneDNN custom operations` messages are harmless informational logs from TensorFlow.

---

## Future Improvements

- **Fisherfaces (LDA)** — class-discriminative subspace (better than PCA for recognition)
- **Deep CNN embeddings** — replace PCA with learned features (FaceNet, ArcFace)
- **Real-time webcam recognition** — OpenCV `VideoCapture` loop
- **GUI** — Tkinter or Gradio interface
- **Data augmentation** — flips, brightness shifts for robustness
- **Cross-validation** — more robust evaluation than a single train/test split

---

## References

1. Turk, M., & Pentland, A. (1991). *Eigenfaces for recognition*. Journal of Cognitive Neuroscience, 3(1), 71–86.
2. Sirovich, L., & Kirby, M. (1987). *Low-dimensional procedure for the characterization of human faces*. JOSA A, 4(3), 519–524.
3. Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
