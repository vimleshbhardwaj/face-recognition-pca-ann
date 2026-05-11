"""
evaluator.py
------------
Handles:
  1. Evaluating the trained ANN on the test set (accuracy, precision, recall, F1).
  2. Plotting the confusion matrix.
  3. Imposter detection via confidence threshold.
  4. Accuracy vs k analysis (train/evaluate for multiple k values).
  5. Displaying sample prediction images.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import seaborn as sns

import tensorflow as tf


# ---------------------------------------------------------------------------
# 1. Standard Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   label_names: list, output_dir: str) -> dict:
    """
    Evaluate ANN on test data and print/save metrics.

    Args:
        model:       Trained Keras model.
        X_test:      Test signatures, shape (n_test, k).
        y_test:      True integer labels, shape (n_test,).
        label_names: Human-readable class names.
        output_dir:  Where to save the confusion matrix plot.

    Returns:
        metrics (dict): accuracy, precision, recall, f1.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0)          # (n_test, n_classes)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 55)
    print("  EVALUATION METRICS")
    print("=" * 55)
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print("=" * 55)
    # Only use labels that actually appear in the test set
    present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    present_names  = [label_names[i] for i in present_labels if i < len(label_names)]

    print("\n  Per-class report:")
    print(classification_report(y_test, y_pred,
                                 labels=present_labels,
                                 target_names=present_names,
                                 zero_division=0))

    # Confusion matrix
    _plot_confusion_matrix(y_test, y_pred, present_names, output_dir, labels=present_labels)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _plot_confusion_matrix(y_true, y_pred, label_names: list,
                            output_dir: str, labels=None) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(label_names)
    fig_size = max(6, min(n, 10))  # cap at 10 inches
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[Evaluator] Confusion matrix saved to: {save_path}")


# ---------------------------------------------------------------------------
# 2. Imposter Detection
# ---------------------------------------------------------------------------

def predict_with_imposter_detection(model, signature: np.ndarray,
                                     label_names: list,
                                     threshold: float = 0.6) -> dict:
    """
    Predict identity with imposter (unknown person) detection.

    If the maximum predicted probability is below `threshold`, the face is
    rejected as an unknown/not-enrolled person.

    Args:
        model        (keras.Model):  Trained ANN.
        signature    (np.ndarray):   Shape (1, k) — PCA projection of query image.
        label_names  (list):         Class names.
        threshold    (float):        Minimum confidence to accept a prediction.

    Returns:
        result (dict): {'identity': str, 'confidence': float, 'is_imposter': bool}
    """
    proba = model.predict(signature, verbose=0)  # (1, n_classes)
    max_conf = np.max(proba)
    pred_idx = np.argmax(proba)

    if max_conf < threshold:
        identity = "Unknown / Not Enrolled"
        is_imposter = True
    else:
        identity = label_names[pred_idx]
        is_imposter = False

    result = {
        "identity": identity,
        "confidence": float(max_conf),
        "is_imposter": is_imposter,
        "all_probabilities": proba[0].tolist()
    }

    status = "⚠ IMPOSTER REJECTED" if is_imposter else "✔ RECOGNISED"
    print(f"\n[Prediction] {status}")
    print(f"             Identity   : {identity}")
    print(f"             Confidence : {max_conf * 100:.1f}%")

    return result


def _load_imposter_signature(image_path: str, mean_face: np.ndarray,
                              feature_matrix: np.ndarray,
                              image_size: tuple = (100, 100)) -> np.ndarray:
    """
    Load a real imposter image from disk, project it through the PCA pipeline,
    and return its face signature ready for ANN prediction.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (image_size[1], image_size[0]))
    flat = img.flatten().astype(np.float64).reshape(-1, 1)   # (mn, 1)
    normalised = flat - mean_face                             # mean zero
    signature = (feature_matrix.T @ normalised).flatten()    # (k,)
    return signature.reshape(1, -1)                           # (1, k)


def run_imposter_demo(model, X_test: np.ndarray, y_test: np.ndarray,
                       label_names: list, output_dir: str,
                       mean_face: np.ndarray = None,
                       feature_matrix: np.ndarray = None,
                       imposters_dir: str = "imposters",
                       threshold: float = 0.6, n_samples: int = 6) -> None:
    """
    Demonstrate genuine recognition vs. imposter rejection.

    Genuine samples  → random picks from the test set.
    Imposter samples → real face images from the `imposters/` folder
                       (people NOT in the training dataset).
                       Falls back to noise vectors if folder is absent.

    Args:
        model:          Trained Keras model.
        X_test:         Test signatures (n_test, k).
        y_test:         True labels (n_test,).
        label_names:    Class name list.
        output_dir:     Where to save the demo plot.
        mean_face:      Mean face vector (mn, 1) — needed for real images.
        feature_matrix: Eigenface directions (mn, k) — needed for real images.
        imposters_dir:  Folder containing real imposter face images.
        threshold:      Confidence threshold for imposter rejection.
        n_samples:      Number of genuine samples to display.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 55)
    print("  IMPOSTER DETECTION DEMO")
    print(f"  Confidence threshold: {threshold}")
    print("=" * 55)

    # ── Genuine predictions from test set ────────────────────────────────────
    print("\n  [Genuine Faces — from test set]")
    genuine_results = []
    sample_indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)),
                                      replace=False)
    for idx in sample_indices:
        sig      = X_test[idx:idx + 1]
        true_name = label_names[y_test[idx]]
        result   = predict_with_imposter_detection(model, sig, label_names, threshold)
        result["true_identity"] = true_name
        genuine_results.append(result)

    # ── Real imposter images (people NOT in training dataset) ─────────────────
    imposter_results = []
    img_extensions   = ('.png', '.jpg', '.jpeg', '.bmp', '.pgm')
    use_real_images  = (os.path.isdir(imposters_dir)
                        and mean_face is not None
                        and feature_matrix is not None)

    if use_real_images:
        imp_files = sorted([
            f for f in os.listdir(imposters_dir)
            if f.lower().endswith(img_extensions)
        ])
        print(f"\n  [Imposter Faces — {len(imp_files)} real image(s) from '{imposters_dir}/' folder]")
        for fname in imp_files:
            fpath = os.path.join(imposters_dir, fname)
            try:
                sig    = _load_imposter_signature(fpath, mean_face, feature_matrix)
                result = predict_with_imposter_detection(model, sig, label_names, threshold)
                result["true_identity"] = f"Unknown ({os.path.splitext(fname)[0]})"
                result["filename"]      = fname
                imposter_results.append(result)
                print(f"     Loaded: {fname}")
            except Exception as e:
                print(f"     [WARN] Skipped {fname}: {e}")
    else:
        # Fallback: random noise vectors
        print("\n  [Imposter Faces — random noise (imposters/ folder not found)]")
        for _ in range(3):
            noise_sig = np.random.randn(1, X_test.shape[1]) * X_test.std() * 3
            result    = predict_with_imposter_detection(model, noise_sig, label_names, threshold)
            result["true_identity"] = "Unknown (Noise)"
            imposter_results.append(result)

    # --- Plot results in a clean grid layout ---
    all_results = genuine_results + imposter_results
    n_genuine   = len(genuine_results)
    n_imp       = len(imposter_results)

    COLS = 3
    genuine_rows = int(np.ceil(n_genuine / COLS))
    imposter_rows = int(np.ceil(n_imp / COLS))
    # 2 headers + genuine rows + imposter rows
    total_rows = genuine_rows + imposter_rows + 2

    fig = plt.figure(figsize=(COLS * 3.5, total_rows * 2.8))
    fig.patch.set_facecolor('#f4f7f9')

    # ── Section header: Genuine Faces ────────────────────────────────────────
    ax_g_header = fig.add_subplot(total_rows, 1, 1)
    ax_g_header.set_facecolor('#000000')  # Solid Black
    ax_g_header.text(0.5, 0.5,
                     f"SECTION A: GENUINE FACE VALIDATION",
                     ha='center', va='center', fontsize=14,
                     fontweight='bold', color='#ffffff')
    ax_g_header.axis('off')

    # ── Genuine face cards ───────────────────────────────────────────────────
    for i, res in enumerate(genuine_results):
        row = i // COLS + 2
        col = i % COLS + 1
        ax = fig.add_subplot(total_rows, COLS, (row - 1) * COLS + col)

        is_correct = res['identity'] == res['true_identity']
        if res['is_imposter']:
            bg, border, status_txt, status_col = '#fffbf0', '#ffa000', 'REJECTED', '#bf360c'
        elif is_correct:
            bg, border, status_txt, status_col = '#f1f8e9', '#2e7d32', 'MATCH',    '#1b5e20'
        else:
            bg, border, status_txt, status_col = '#fff1f1', '#d32f2f', 'MISMATCH', '#b71c1c'

        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(1.5)

        conf = res['confidence'] * 100
        text = (f"● {status_txt} ●\n"
                f"PRED: {res['identity'].upper()}\n"
                f"TRUE: {res['true_identity'].upper()}\n"
                f"CONF: {conf:.1f}%")
        ax.text(0.5, 0.58, text, ha='center', va='center',
                fontsize=9, color=status_col, fontweight='bold',
                linespacing=1.5)

        # Progress Bar (Confidence)
        bar_bg = ax.inset_axes([0.15, 0.12, 0.7, 0.08])
        bar_bg.barh(0, 100, color='#e0e0e0', height=1.0)
        bar_bg.barh(0, conf, color=border, height=1.0)
        bar_bg.set_xlim(0, 100)
        bar_bg.axis('off')

        ax.axis('off')

    # ── Section header: Imposters ─────────────────────────────────────────────
    imp_header_row = genuine_rows + 2
    ax_i_header = fig.add_subplot(total_rows, 1, imp_header_row)
    ax_i_header.set_facecolor('#000000')  # Solid Black
    ax_i_header.text(0.5, 0.5,
                     f"SECTION B: IMPOSTER DETECTION",
                     ha='center', va='center', fontsize=14,
                     fontweight='bold', color='#ffffff')
    ax_i_header.axis('off')

    # ── Imposter cards ───────────────────────────────────────────────────────
    imp_start_row = genuine_rows + 3
    for i, res in enumerate(imposter_results):
        row = imp_start_row + i // COLS
        col = i % COLS + 1
        ax = fig.add_subplot(total_rows, COLS, (row - 1) * COLS + col)

        rejected = res['is_imposter']
        bg     = '#fff1f1' if not rejected else '#f3e5f5'
        border = '#d32f2f' if not rejected else '#4a148c'
        status_txt = 'FAILED (False Pos)' if not rejected else 'SUCCESS (Rejected)'
        status_col = '#b71c1c' if not rejected else '#4a148c'

        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(1.5)

        conf = res['confidence'] * 100
        text = (f"● {status_txt} ●\n"
                f"PRED: {res['identity'].upper()}\n"
                f"TRUE: UNKNOWN\n"
                f"CONF: {conf:.1f}%")
        ax.text(0.5, 0.58, text, ha='center', va='center',
                fontsize=9, color=status_col, fontweight='bold',
                linespacing=1.5)

        # Progress Bar (Confidence)
        bar_bg = ax.inset_axes([0.15, 0.12, 0.7, 0.08])
        bar_bg.barh(0, 100, color='#e0e0e0', height=1.0)
        bar_bg.barh(0, conf, color=border, height=1.0)
        bar_bg.set_xlim(0, 100)
        bar_bg.axis('off')

        ax.axis('off')

    # ── Final refinement ─────────────────────────────────────────────────────
    plt.subplots_adjust(hspace=0.5, wspace=0.35, top=0.92, bottom=0.05)
    save_path = os.path.join(output_dir, "imposter_detection_demo.png")
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"\n[Evaluator] Imposter demo saved to: {save_path}")


# ---------------------------------------------------------------------------
# 3. Accuracy vs k Analysis
# ---------------------------------------------------------------------------

def accuracy_vs_k(face_db: np.ndarray, labels: np.ndarray,
                  mean_face: np.ndarray, eigenface_dirs: np.ndarray,
                  label_names: list, k_values: list,
                  output_dir: str, model_dir: str,
                  epochs: int = 30, batch_size: int = 16) -> dict:
    """
    Train and evaluate the ANN for each k value in k_values.
    Plot accuracy vs k graph.

    Args:
        face_db        (np.ndarray): (mn, p) raw face database.
        labels         (np.ndarray): Integer labels (p,).
        mean_face      (np.ndarray): (mn, 1) mean face.
        eigenface_dirs (np.ndarray): (mn, p) all eigenface directions.
        label_names    (list):       Class names.
        k_values       (list):       List of k values to evaluate.
        output_dir     (str):        Where to save plot.
        model_dir      (str):        Where to save models (per k).
        epochs, batch_size: Training hyperparameters.

    Returns:
        k_accuracy (dict): {k: accuracy} mapping.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    os.makedirs(output_dir, exist_ok=True)

    delta = face_db - mean_face
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    num_classes = len(label_names)

    k_accuracy = {}

    print("\n" + "=" * 55)
    print("  ACCURACY vs K ANALYSIS")
    print("=" * 55)

    for k in k_values:
        # Clip k to number of available eigenvectors
        k_actual = min(k, eigenface_dirs.shape[1])
        feature_matrix = eigenface_dirs[:, :k_actual]
        signatures = (feature_matrix.T @ delta).T  # (p, k_actual)

        X_tr, X_te, y_tr, y_te = train_test_split(
            signatures, encoded_labels, test_size=0.4,
            random_state=42, stratify=encoded_labels
        )

        # Build a fresh ANN for each k
        from src.ann_model import build_ann
        model = build_ann(k_actual, num_classes)

        model.fit(X_tr, y_tr,
                  validation_data=(X_te, y_te),
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0)

        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc = accuracy_score(y_te, y_pred)
        k_accuracy[k] = acc
        print(f"  k = {k:3d}  →  Accuracy = {acc * 100:.2f}%")

    # --- Plot ---
    ks = list(k_accuracy.keys())
    accs = [k_accuracy[k] * 100 for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs, 'bo-', linewidth=2, markersize=8)
    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.1f}%", (k, a), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_title("Classification Accuracy vs Number of Eigenfaces (k)",
                 fontweight='bold', fontsize=13)
    ax.set_xlabel("k (Number of Principal Components)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    # Add analysis annotation
    best_k = ks[np.argmax(accs)]
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.6, label=f'Best k = {best_k}')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_vs_k.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n[Evaluator] Accuracy vs k plot saved to: {save_path}")
    print(f"            Best k = {best_k}  with accuracy = {k_accuracy[best_k] * 100:.2f}%")

    # Analysis explanation
    print("\n  Analysis:")
    print("  • Small k: Too few features → under-representation → low accuracy.")
    print("  • Optimal k: Enough variance captured for good discrimination.")
    print("  • Large k: Noise components included → possible overfitting.")
    print("  • Computational cost grows with k (more input neurons + projections).")

    return k_accuracy
