"""
app.py
------
Streamlit Web App for PCA-ANN Face Recognition System.
Loads pre-trained model artifacts and provides a live prediction interface.
"""

import streamlit as st
import numpy as np
import cv2
import json
import os
import sys
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition | PCA + ANN",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e8eaf6;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.9);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-4px); }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Result boxes */
    .result-match {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-reject {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.1));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-name {
        font-size: 2rem;
        font-weight: 700;
        margin: 8px 0;
    }

    .result-conf {
        font-size: 1.1rem;
        color: #94a3b8;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(167,139,250,0.4);
        border-radius: 16px;
        padding: 20px;
        background: rgba(255,255,255,0.03);
    }

    /* Section title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e8eaf6;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(167,139,250,0.4);
    }

    /* Confidence bar */
    .conf-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 999px;
        height: 10px;
        margin: 8px 0 4px;
        overflow: hidden;
    }

    .conf-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        transition: width 0.5s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124,58,237,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    """Load all pre-trained model artifacts once and cache them."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load

    model_dir = "models"
    config_path = os.path.join(model_dir, "train_config.json")

    if not os.path.exists(config_path):
        return None, None, None, None, None

    with open(config_path, "r") as f:
        cfg = json.load(f)

    feature_matrix = np.load(os.path.join(model_dir, "feature_matrix.npy"))
    mean_face = np.load(os.path.join(model_dir, "mean_face.npy"))
    label_names = cfg["label_names"]
    threshold = cfg["confidence_threshold"]
    image_size = tuple(cfg["image_size"])

    model_path = os.path.join(model_dir, "face_recognition_ann.h5")
    model = keras_load(model_path, compile=False)

    return model, feature_matrix, mean_face, label_names, threshold, image_size, cfg


def predict_face(pil_image, model, feature_matrix, mean_face, label_names,
                 threshold, image_size):
    """Run full PCA + ANN inference on a PIL image."""
    # Convert to grayscale numpy array
    img_np = np.array(pil_image.convert("L"))
    img_resized = cv2.resize(img_np, (image_size[1], image_size[0]))
    flat = img_resized.flatten().astype(np.float64)

    # Mean normalise + PCA project
    normalised = flat.reshape(-1, 1) - mean_face        # (mn, 1)
    signature = (feature_matrix.T @ normalised).flatten()  # (k,)
    signature = signature.reshape(1, -1)                 # (1, k)

    # ANN predict
    proba = model.predict(signature, verbose=0)[0]       # (n_classes,)
    max_conf = float(np.max(proba))
    pred_idx = int(np.argmax(proba))

    is_imposter = max_conf < threshold
    identity = "Unknown / Not Enrolled" if is_imposter else label_names[pred_idx]

    return {
        "identity": identity,
        "confidence": max_conf,
        "is_imposter": is_imposter,
        "all_proba": proba.tolist(),
        "label_names": label_names,
    }


def draw_confidence_bar(confidence, color="#a78bfa"):
    pct = int(confidence * 100)
    st.markdown(f"""
    <div class="conf-bar-container">
      <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <p style="text-align:center; color:#94a3b8; font-size:0.9rem;">{pct}% confidence</p>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Face Recognition")
    st.markdown("**PCA + ANN Pipeline**")
    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🔍 Predict — Upload & identify a face  
    - 📊 Results — View evaluation metrics  
    - ℹ️ About — Project details  
    """)
    st.markdown("---")

    # Load artifacts here for sidebar info
    artifacts = load_artifacts()
    if artifacts[0] is not None:
        model, feature_matrix, mean_face, label_names, threshold, image_size, cfg = artifacts
        st.markdown("### 📋 Model Info")
        st.markdown(f"**Classes:** {len(label_names)}")
        st.markdown(f"**PCA Components (k):** {cfg['k']}")
        st.markdown(f"**Image Size:** {image_size[0]}×{image_size[1]}")
        st.markdown(f"**Threshold:** {threshold}")
        st.markdown("**Enrolled Identities:**")
        enrolled = [n for n in label_names if n not in ("Iris", "faces")]
        for name in enrolled:
            st.markdown(f"  - {name}")
        st.markdown("---")
        st.success("✅ Model loaded & ready")
    else:
        st.error("❌ Model artifacts not found. Run `python train.py` first.")


# ─── Tab Navigation ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Results", "ℹ️ About"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 🔍 Face Recognition — Live Prediction")
    st.markdown("Upload a face image to identify the person using the **PCA + ANN** model.")
    st.markdown("---")

    if artifacts[0] is None:
        st.error("Model artifacts not found. Please run `python train.py` first.")
        st.stop()

    model, feature_matrix, mean_face, label_names, threshold, image_size, cfg = artifacts

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a face image (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp", "pgm"],
            label_visibility="collapsed"
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

            # Preview as grayscale (how the model sees it)
            gray = pil_img.convert("L").resize((image_size[1], image_size[0]))
            st.image(gray, caption=f"Model Input ({image_size[0]}×{image_size[1]} grayscale)",
                     use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Running PCA + ANN inference…"):
                result = predict_face(pil_img, model, feature_matrix, mean_face,
                                      label_names, threshold, image_size)

            if result["is_imposter"]:
                st.markdown(f"""
                <div class="result-reject">
                    <div style="font-size:3rem;">⚠️</div>
                    <div style="color:#ef4444; font-size:1rem; font-weight:600; text-transform:uppercase; letter-spacing:2px;">
                        IMPOSTER REJECTED
                    </div>
                    <div class="result-name" style="color:#fca5a5;">Unknown / Not Enrolled</div>
                    <div class="result-conf">Confidence below threshold ({int(threshold*100)}%)</div>
                </div>
                """, unsafe_allow_html=True)
                draw_confidence_bar(result["confidence"], "#ef4444")
            else:
                st.markdown(f"""
                <div class="result-match">
                    <div style="font-size:3rem;">✅</div>
                    <div style="color:#10b981; font-size:1rem; font-weight:600; text-transform:uppercase; letter-spacing:2px;">
                        IDENTITY RECOGNISED
                    </div>
                    <div class="result-name" style="color:#6ee7b7;">{result['identity']}</div>
                    <div class="result-conf">{result['confidence']*100:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
                draw_confidence_bar(result["confidence"], "#10b981")

            # ── All-class probability breakdown ──────────────────────────────
            st.markdown("#### 📊 Probability Breakdown")
            proba_arr = np.array(result["all_proba"])
            sorted_idx = np.argsort(proba_arr)[::-1][:5]  # top 5

            for idx in sorted_idx:
                name = label_names[idx]
                prob = proba_arr[idx]
                bar_color = "#10b981" if not result["is_imposter"] and name == result["identity"] else "#a78bfa"
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.markdown(f"**{name}**")
                with col_bar:
                    pct = int(prob * 100)
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.1);border-radius:999px;height:8px;overflow:hidden;margin-top:8px;">
                      <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:999px;"></div>
                    </div>
                    <small style="color:#94a3b8;">{pct}%</small>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:80px 20px; color:#64748b;">
                <div style="font-size:4rem;">🖼️</div>
                <div style="font-size:1.1rem; margin-top:12px;">Upload an image to see prediction</div>
                <div style="font-size:0.85rem; margin-top:8px; color:#475569;">
                    Supported: JPG, PNG, BMP, PGM
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Threshold control ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Advanced: Adjust Confidence Threshold"):
        new_threshold = st.slider(
            "Imposter Rejection Threshold",
            min_value=0.0, max_value=1.0,
            value=float(threshold), step=0.05,
            help="Faces with confidence below this are rejected as unknown."
        )
        if uploaded and new_threshold != threshold:
            with st.spinner("Re-running with new threshold…"):
                result2 = predict_face(pil_img, model, feature_matrix, mean_face,
                                       label_names, new_threshold, image_size)
            st.info(f"With threshold {new_threshold:.2f}: **{result2['identity']}** "
                    f"({result2['confidence']*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("# 📊 Evaluation Results")
    st.markdown("Pre-generated outputs from the full evaluation pipeline.")
    st.markdown("---")

    output_dir = "outputs"

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("### 🏆 Model Performance (Test Split — 40%)")

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("57.22%", "Accuracy"),
        ("57.05%", "Precision"),
        ("57.22%", "Recall"),
        ("56.81%", "F1-Score"),
    ]
    for col, (val, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Plots ─────────────────────────────────────────────────────────────────
    def show_output_image(filename, caption, col=None):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            img = Image.open(path)
            target = col if col else st
            target.image(img, caption=caption, use_container_width=True)
        else:
            target = col if col else st
            target.warning(f"`{filename}` not found. Run `python test.py` to generate it.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔥 Confusion Matrix")
        show_output_image("confusion_matrix.png", "Confusion Matrix — Test Set", col_a)
    with col_b:
        st.markdown("#### 📈 Training History")
        show_output_image("training_history_k30.png", "Loss & Accuracy over Epochs", col_b)

    st.markdown("---")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 👤 Eigenfaces Grid")
        show_output_image("eigenfaces_grid.png", "Top-30 Eigenfaces (Principal Components)", col_c)
    with col_d:
        st.markdown("#### 📉 Eigenvalue Distribution")
        show_output_image("eigenvalue_distribution.png", "Scree Plot — Variance Explained", col_d)

    st.markdown("---")

    st.markdown("#### 🕵️ Imposter Detection Demo")
    show_output_image("imposter_detection_demo.png", "Genuine vs Imposter Prediction Results")

    st.markdown("---")
    st.markdown("#### 🔍 Accuracy vs k (PCA Components)")
    show_output_image("accuracy_vs_k.png", "How accuracy changes with number of eigenfaces")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        ## 🎭 PCA + ANN Face Recognition System

        This project implements a **classical machine learning** approach to face recognition
        using **Principal Component Analysis (PCA)** for dimensionality reduction and an
        **Artificial Neural Network (ANN)** for classification.

        ### 🧠 How It Works

        1. **Data Loading** — Face images are loaded, converted to grayscale, and resized to 100×100
        2. **Mean Normalisation** — The mean face is subtracted from each image
        3. **PCA (Eigenfaces)** — Using the surrogate covariance trick (Turk & Pentland, 1991),
           we compute eigenfaces without the O(mn²) cost
        4. **Feature Projection** — Each face is projected onto the top-k=30 eigenfaces to get a
           compact 30-dimensional signature
        5. **ANN Classification** — A 3-layer neural network classifies the signature:
           `Input(30) → Dense(128) → Dropout → Dense(64) → Dropout → Softmax(11)`
        6. **Imposter Rejection** — If max class probability < 0.6, face is rejected as unknown

        ### 📐 Mathematical Foundation

        The key insight is the **surrogate covariance matrix**:
        - True covariance: **mn × mn** → 10,000 × 10,000 = 100M entries (infeasible)
        - Surrogate: **p × p** → 450 × 450 = 202K entries (efficient!)
        - True eigenfaces recovered via: `u_i = Δ × v_i` (then normalised)

        ### 📊 Dataset
        - **9 Bollywood celebrities** — Aamir, Ajay, Akshay, Alia, Amitabh,
          Deepika, Disha, Farhan, Ileana
        - **50 images each** (450 total)
        - 60% train / 40% test split
        """)

    with col_right:
        st.markdown("### 🗂️ Architecture")
        st.markdown("""
        <div class="metric-card" style="text-align:left;">
            <code style="color:#a78bfa;">
            Input (30 dims)<br>
            &nbsp;&nbsp;↓<br>
            Dense(128, ReLU)<br>
            BatchNorm + Dropout<br>
            &nbsp;&nbsp;↓<br>
            Dense(64, ReLU)<br>
            BatchNorm + Dropout<br>
            &nbsp;&nbsp;↓<br>
            Dense(11, Softmax)<br>
            </code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Hyperparameters")
        params = {
            "PCA k": 30, "Image Size": "100×100", "Threshold": 0.6,
            "Epochs": 50, "Batch Size": 16, "Dropout": 0.3,
            "Optimizer": "Adam", "Loss": "Sparse CE"
        }
        for k, v in params.items():
            st.markdown(f"**{k}:** `{v}`")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        Built with Python • PCA • TensorFlow/Keras • Streamlit<br>
        Turk & Pentland (1991) — Eigenfaces for Recognition
    </div>
    """, unsafe_allow_html=True)
