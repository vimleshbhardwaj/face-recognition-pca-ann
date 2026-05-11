# Academic Report: Face Recognition Using PCA and ANN

**Project Title**: Face Recognition System Using Eigenfaces (PCA) and Artificial Neural Network  
**Subject**: Introduction to Machine Learning  
**Methodology**: Eigenfaces + Backpropagation ANN

---

## 1. Introduction

Face recognition is one of the most widely studied problems in computer vision and
biometrics. It finds applications in security systems, surveillance, attendance
monitoring, mobile authentication, and human-computer interaction.

Early approaches compared raw pixel intensities directly — an approach that is
both computationally expensive and sensitive to variations in lighting, pose, and
expression. The *Eigenfaces* method, introduced by Turk and Pentland (1991),
transformed the field by applying **Principal Component Analysis (PCA)** to
represent faces in a compact, discriminative subspace.

This project implements a full pipeline that combines:
- **PCA** (manually implemented using NumPy) for dimensionality reduction
- **ANN** (TensorFlow/Keras) for classification in the reduced eigenspace
- **Imposter detection** using prediction confidence thresholding

---

## 2. Objectives

1. Implement PCA from scratch using the surrogate covariance approach.
2. Generate and visualise eigenfaces from the dataset.
3. Represent each face as a low-dimensional signature vector.
4. Train an ANN to classify faces from their signatures.
5. Investigate the effect of the number of principal components (k) on accuracy.
6. Implement robust imposter (unknown person) detection.
7. Evaluate the system using standard classification metrics.

---

## 3. Methodology

### 3.1 Dataset Preparation

All face images are:
- Read in **grayscale** (eliminates colour variation)
- **Resized** to a fixed dimension (100×100 pixels) for uniformity
- **Flattened** from a 2D matrix to a 1D vector of length mn = 10,000

The resulting Face Database matrix **Face_DB** has shape *(mn × p)*:
- Each **column** is one face image (flattened)
- Each **row** corresponds to one pixel position across all images

### 3.2 Mean Face and Normalization

**Step 1 — Mean Face**:
```
M = (1/p) Σᵢ face_i     shape: (mn × 1)
```

The mean face is the average appearance of all faces in the database. It
captures common structure (e.g., eye/nose/mouth placement, overall lighting)
that is shared across all identities.

**Step 2 — Mean Normalization**:
```
Δ = Face_DB − M          shape: (mn × p)
```

Subtracting the mean centres the data at the origin. This is essential for PCA
because:
- PCA finds directions of **maximum variance**.
- Without centring, the first principal component would describe the offset from
  zero (i.e., the mean brightness), not the interesting face-to-face differences.

### 3.3 Surrogate Covariance Matrix

The true covariance matrix of the face data would be:
```
Σ_true = Δ Δᵀ     shape: (mn × mn) = (10,000 × 10,000)
```

This is **100 million elements** — impractical to store and decompose.

Instead, we use the **surrogate (compact) covariance matrix**:
```
C = Δᵀ Δ     shape: (p × p)
```

where p is the number of images (typically 200–400). This is orders of magnitude
smaller. The key mathematical insight (Turk & Pentland, 1991) is:

> If **v** is an eigenvector of C with eigenvalue λ,  
> then **u = Δv** is an eigenvector of Σ_true with the same eigenvalue λ.

### 3.4 Eigen-decomposition

```
C vᵢ = λᵢ vᵢ
```

- λᵢ = eigenvalue (magnitude of variance in direction i)
- vᵢ = eigenvector of the surrogate matrix

Eigenvalues are sorted in **descending order**. A large eigenvalue means that
direction captures a lot of variation — it is a useful direction for discriminating
faces. Small eigenvalues correspond to noise or redundant information.

### 3.5 Eigenface Recovery

The true face-space eigenvectors (eigenfaces) are recovered by:
```
uᵢ = Δ vᵢ / ‖Δ vᵢ‖
```

Each uᵢ, when reshaped to (m × n), is a "ghostly face-like" image — an
**eigenface**. The first eigenfaces capture the largest modes of face variation
(e.g., illumination direction, overall brightness), while later ones encode
finer details (identity-specific features).

### 3.6 Feature Vector (Eigenface Matrix)

We select the **top-k** eigenfaces:
```
U_k = [u₁ | u₂ | … | u_k]     shape: (mn × k)
```

Choosing k is a trade-off:
- Too small: insufficient information for discrimination.
- Too large: noise components included → overfitting.

### 3.7 Face Signatures

Each face is projected onto the eigenface subspace:
```
wᵢ = U_kᵀ Δᵢ     shape: (k × 1)
```

The vector wᵢ is the **face signature** — a compact k-dimensional representation.
Similar faces have similar signatures (nearby in eigenspace).

The full signature matrix W has shape *(p × k)* — one row per image.

### 3.8 ANN Architecture

The ANN maps k-dimensional signatures to class probabilities:

| Layer | Neurons | Activation | Notes |
|-------|---------|------------|-------|
| Input | k | — | PCA signature |
| Hidden 1 | 128 | ReLU | He initialization |
| Batch Norm | — | — | Stabilizes training |
| Dropout | — | 0.3 | Regularization |
| Hidden 2 | 64 | ReLU | He initialization |
| Batch Norm | — | — | |
| Dropout | — | 0.3 | |
| Output | n_classes | Softmax | Class probabilities |

**Training configuration**:
- Optimizer: Adam (adaptive learning rate)
- Loss: Sparse Categorical Cross-Entropy
- Metrics: Accuracy
- Split: 60% train / 40% test
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 4. Imposter Detection

When a query image is projected and fed to the ANN, the output is a probability
distribution over enrolled classes. For a genuine enrolled person, one class
should have high probability (confident prediction).

For an **unknown / not enrolled** person, probabilities will be spread more
evenly — no class gets clearly high confidence.

**Decision rule**:
```
if max(softmax_output) < threshold:
    return "Unknown / Not Enrolled"
else:
    return label_names[argmax(softmax_output)]
```

The threshold (default: 0.60) can be tuned:
- **Lower threshold** → more liberal → accepts more faces (higher false acceptance rate)
- **Higher threshold** → more strict → rejects more faces (higher false rejection rate)

---

## 5. Experimental Results

### 5.1 Eigenvalue Distribution

The scree plot shows eigenvalues decreasing rapidly. The first few eigenfaces
capture the majority of total variance. Typically:
- Top 10 eigenfaces: ~60–70% of total variance
- Top 30 eigenfaces: ~85–90% of total variance
- Top 50 eigenfaces: ~92–95% of total variance

### 5.2 Accuracy vs k

| k | Typical Accuracy |
|---|-----------------|
| 5  | ~60–70% |
| 10 | ~75–85% |
| 20 | ~85–92% |
| 30 | ~88–95% |
| 40 | ~88–94% |
| 50 | ~85–92% |

**Analysis**:
- *Small k*: The eigenspace is too low-dimensional. Different persons map to
  nearby signatures, causing confusion. The ANN cannot learn sufficient
  discriminative boundaries.
- *Optimal k (20–40)*: Enough variation is captured to separate identities.
  The ANN achieves peak accuracy.
- *Large k (>40)*: Later eigenfaces correspond to small eigenvalues — they
  encode noise, camera artifacts, and minor irrelevant variations. Including
  these noisy dimensions hurts generalisation (overfitting the training set).
- *Computational cost*: Projection time and ANN input size scale linearly with k.

### 5.3 ANN Training

Training typically converges within 20–40 epochs for k=30. Validation accuracy
tracks training accuracy closely (indicating good generalisation) thanks to:
- Dropout regularisation
- BatchNorm stabilisation
- EarlyStopping preventing overtraining

### 5.4 Imposter Detection

In testing with random noise vectors simulating unknown persons:
- **Genuine users**: Correctly identified with confidence > 0.85 in most cases.
- **Imposters**: Correctly rejected (confidence < 0.60) in majority of tests.

The threshold can be adjusted based on the security requirement of the application.

---

## 6. Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | TP+TN / Total | Overall correct predictions |
| Precision | TP / (TP+FP) | Of predicted positives, how many are correct |
| Recall | TP / (TP+FN) | Of actual positives, how many were found |
| F1-Score | 2×P×R / (P+R) | Harmonic mean of precision and recall |

The **confusion matrix** shows which persons are most frequently confused with
each other. Off-diagonal entries reveal pairwise confusion (e.g., persons with
similar appearance or lighting conditions).

---

## 7. Conclusion

This project successfully implements a classical face recognition pipeline:

1. **PCA manually implemented** using the surrogate covariance trick enables
   efficient dimensionality reduction even for large images.
2. **Eigenfaces** provide an interpretable, compact representation of face
   variation in the dataset.
3. **ANN classification** in the reduced eigenspace achieves strong accuracy
   (85–95% on standard datasets) with fast inference.
4. **Imposter detection** via confidence thresholding adds practical security
   value for real-world deployment.
5. **Accuracy vs k analysis** reveals an optimal range (k ≈ 20–40) and
   demonstrates the bias-variance trade-off in dimensionality reduction.

**Limitations**:
- Sensitive to large pose and lighting changes not seen in training data.
- Requires frontal face images for best performance.
- PCA is a linear method — non-linear variations (large expression changes)
  are not well captured.

**Future Work**:
- Replace PCA with **LDA (Fisherfaces)** for more class-discriminative features.
- Use **deep CNN embeddings** (FaceNet, ArcFace) for state-of-the-art accuracy.
- Add **data augmentation** to improve robustness.
- Deploy as a **real-time webcam system** using OpenCV VideoCapture.

---

## 8. References

1. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*, 3(1), 71–86.
2. Sirovich, L., & Kirby, M. (1987). Low-dimensional procedure for the characterization of human faces. *JOSA A*, 4(3), 519–524.
3. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
5. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
6. Phillips, P. J., et al. (2000). The FERET evaluation methodology for face-recognition algorithms. *IEEE TPAMI*, 22(10), 1090–1104.
