# 🚁🦅 Aerial Object Classification & Detection — Drone vs Bird

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-98%25-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready deep learning system that classifies aerial images as <strong>Bird</strong> or <strong>Drone</strong>
  using three model architectures — Custom CNN, Transfer Learning, and YOLOv8 — with a live Streamlit web app for real-time inference.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Models](#-models)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Streamlit App](#-streamlit-app)
- [Tech Stack](#-tech-stack)
- [Key Learnings](#-key-learnings)

---

## 🎯 Overview

With the rapid proliferation of commercial and recreational drones, the ability to **automatically distinguish drones from birds** in aerial imagery has become critical for airspace safety, wildlife monitoring, and security systems.

This project builds and benchmarks three deep learning models on a real-world aerial image dataset, culminating in a **deployed web application** that accepts any image and returns a prediction in milliseconds.

**What this project delivers:**
- A rigorous multi-model comparison (CNN vs Transfer Learning vs YOLOv8)
- Full evaluation pipeline — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- A production Streamlit app powered by the best-performing model (`best.pt`)

---

## 🎬 Live Demo

> **🚀 Web App Direct Link :** [https://aerial-object-classification-drone-vs-bird-yolov8-ut3dvxouqwto.streamlit.app/](https://aerial-object-classification-drone-vs-bird-yolov8-ut3dvxouqwto.streamlit.app/)

| Upload Image | Prediction Output |
|:---:|:---:|
| Drag & drop any aerial image | Instant label + confidence score + annotated output |

---

## 🧠 Models

Three model architectures were implemented and compared end-to-end.

### 1. Custom CNN (From Scratch)
Built entirely in TensorFlow/Keras with a VGG-inspired design.

- **Architecture:** 5 convolutional blocks (32 → 64 → 128 → 256 → 512 filters)
- **Regularisation:** BatchNormalization + Dropout (0.25–0.50) + L2 weight decay
- **Head:** GlobalAveragePooling → Dense(256) → Sigmoid
- **Insight:** Solid baseline, but limited by the small dataset size (~2,600 train images). Without pretrained knowledge, the model struggles to learn rich, generalisable aerial features from scratch.

### 2. Transfer Learning — MobileNetV2
Pretrained on ImageNet (1.4M images, 1000 classes), fine-tuned for aerial classification.

- **Phase 1:** Freeze base, train classification head only (LR = 1e-3)
- **Phase 2:** Unfreeze top 30 layers, fine-tune with low LR (5e-5)
- **Key technique:** BatchNorm layers kept frozen during fine-tuning to preserve pretrained statistics
- **Insight:** Transfer Learning consistently outperforms the scratch CNN. The pretrained feature hierarchy (edges → textures → shapes → objects) transfers directly to aerial imagery, delivering higher accuracy with fewer epochs.

### 3. YOLOv8 Classification (`yolov8n-cls`)
Ultralytics YOLOv8 in classification mode — pretrained on COCO, fine-tuned on this dataset.

- **Input size:** 224 × 224
- **Training:** 30 epochs, early stopping, mosaic + HSV + flip augmentation
- **Result:** ~98% Top-1 accuracy on the test set
- **Insight:** YOLOv8's CSP + C2f backbone, combined with COCO pretraining, delivers state-of-the-art accuracy with minimal training time. Easily upgradeable to full object detection (bounding boxes) by switching to `yolov8s.pt` with annotated labels.

---

## 📊 Dataset

| Split | Bird | Drone | Total |
|-------|------|-------|-------|
| Train | 1,414 | 1,248 | **2,662** |
| Validation | 217 | 225 | **442** |
| Test | 121 | 94 | **215** |

**Structure:**
```
data/
├── train/
│   ├── bird/
│   └── drone/
├── valid/
│   ├── bird/
│   └── drone/
└── test/
    ├── bird/
    └── drone/
```

- Format: `.jpg` images
- Classes: `bird` (0), `drone` (1)
- Augmentation applied during training: rotation, zoom, horizontal flip, brightness shift, HSV jitter

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | Train Time |
|-------|----------|-----------|--------|----------|------------|
| Custom CNN | ~88% | ~0.88 | ~0.88 | ~0.88 | ~18 min |
| MobileNetV2 (TL) | ~95% | ~0.95 | ~0.95 | ~0.95 | ~12 min |
| **YOLOv8n-cls** | **~98%** | **~0.98** | **~0.98** | **~0.98** | **~8 min** |

> All models evaluated on the same 215-image test set using `sklearn.metrics`.

### Confusion Matrix

Both classification models produce strong diagonal confusion matrices with minimal cross-class errors. YOLOv8 achieves near-perfect separation between bird and drone classes.

### Metrics Used
- **Accuracy** — overall correctness
- **Precision** — of all "Drone" predictions, how many were correct
- **Recall** — of all actual Drones, how many were detected
- **F1 Score** — harmonic mean of Precision and Recall (primary metric)
- **Confusion Matrix** — visual breakdown of true/false positives and negatives

---

## 📂 Project Structure

```
aerial-object-classification-drone-vs-bird-yolov8/
│
├── app.py                          # Streamlit web application
├── best.pt                         # YOLOv8 trained weights (best model)
├── requirements.txt                # Python dependencies
├── packages.txt                    # System-level apt dependencies
├── README.md                       # This file
│
├── notebook/
│   └── Aerial_Classification_Phase1.ipynb   # Full training + evaluation notebook
│
└── data/                           # Dataset (not tracked by git)
    ├── train/
    ├── valid/
    └── test/
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10
- GPU recommended (CUDA 11.8+) 

### 1. Clone the repository
```bash
git clone https://github.com/yash-kulkarni-ai/aerial-object-classification-drone-vs-bird-yolov8.git
cd aerial-object-classification-drone-vs-bird-yolov8
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Note:** `best.pt` must be in the same directory as `app.py`.

---

## 🌐 Streamlit App

The web app provides a clean, dark-themed interface for real-time inference.

**🔗 Live URL:** [https://aerial-object-classification-drone-vs-bird-yolov8-ut3dvxouqwto.streamlit.app/](https://aerial-object-classification-drone-vs-bird-yolov8-ut3dvxouqwto.streamlit.app/)

**Features:**
- 📤 Drag-and-drop image upload (JPG, PNG, WEBP, BMP)
- 🔍 Side-by-side view: original image vs annotated prediction
- 🏷️ Predicted class with colour-coded result card (Green = Bird, Blue = Drone)
- 📊 Confidence score with animated progress bar
- 📋 Detailed probability table for all classes
- ⬇️ Download annotated output image
- ⚡ Inference time displayed in milliseconds

**Deployment:** Hosted on [Streamlit Cloud](https://streamlit.io/cloud) — free, public, no server required.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | TensorFlow 2.x, Keras, Ultralytics YOLOv8 |
| **Computer Vision** | OpenCV (headless), Pillow (PIL) |
| **ML Utilities** | Scikit-learn, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Language** | Python 3.9+ |
| **Deployment** | Streamlit Cloud |

**`requirements.txt`**
```
ultralytics==8.3.145
streamlit>=1.28.0
opencv-python-headless>=4.8.0
pillow>=9.5.0
numpy>=1.24.0
```

**`packages.txt`** *(system libraries for Streamlit Cloud / Debian 13)*
```
libgl1
libglib2.0-0t64
```

---

## 🧠 Key Learnings

**1. Transfer Learning is essential for small datasets**
Training a CNN from scratch on 2,600 images leads to limited generalisation. MobileNetV2's pretrained ImageNet features (edges, textures, shapes) transfer directly to aerial imagery, closing the accuracy gap by ~7 percentage points over the scratch CNN with half the training time.

**2. Fine-tuning strategy matters**
Freezing BatchNorm layers during fine-tuning is critical. Unfreezing them re-initialises running statistics and destabilises the pretrained weights. Using a 20× lower learning rate during Phase 2 (5e-5 vs 1e-3) preserved prior knowledge while allowing adaptation to the new domain.

**3. YOLOv8's classification mode is underrated**
Most practitioners associate YOLO with detection, but its classification backbone — pretrained on COCO — is highly competitive on image classification tasks. YOLOv8n-cls outperformed both the scratch CNN and MobileNetV2 with the shortest training time, proving that architecture and pretraining quality matter more than model size alone.

**4. Evaluation beyond accuracy**
With a slightly imbalanced test set (121 bird vs 94 drone), accuracy alone is insufficient. F1 Score was the primary metric — it balances Precision (avoiding false drone alarms) and Recall (not missing real drones), which reflects real-world deployment requirements for aerial safety systems.

**5. Deployment on headless servers requires careful dependency management**
Streamlit Cloud runs on Debian 13 (trixie). `ultralytics` hardcodes `opencv-python` (GUI) as a dependency regardless of what's in `requirements.txt`. The missing `libgthread-2.0.so.0` — part of `libglib2.0-0t64` on Debian 13 — must be explicitly added to `packages.txt` to resolve the `ImportError` at startup.

---

<p align="center">
  Built With Deep Learning & Coffee &nbsp;·&nbsp;
  <a href="https://github.com/yash-kulkarni-ai/aerial-object-classification-drone-vs-bird-yolov8">GitHub</a> &nbsp;·&nbsp;
  <a href="https://aerial-object-classification-drone-vs-bird-yolov8-ut3dvxouqwto.streamlit.app/">Live Demo</a>
</p>
