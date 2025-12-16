# Face Recognition and Dimensionality Reduction: PCA vs LDA with Classical ML & Deep Learning

## 📌 Overview

This project presents a **comprehensive face recognition study** using the **ORL Face Dataset**, combining:

- **Classical machine learning classifiers** with  
  **PCA (Principal Component Analysis)** and  
  **LDA (Linear Discriminant Analysis)** for dimensionality reduction
- A **deep learning–based CNN model** implemented in **PyTorch**
- Extensive **performance evaluation, visualization, and comparison**

The goal is to analyze how **feature extraction techniques** impact classification performance and to contrast **traditional ML pipelines** with **end-to-end deep learning** approaches.

---

## 🧠 Methodology

### 🔹 Classical ML Pipeline
1. Image preprocessing (grayscale, resizing, normalization)
2. Feature extraction using:
   - **PCA** (variance-based dimensionality reduction)
   - **LDA** (class-discriminative projection)
3. Classification using:
   - Support Vector Machine (SVM)
   - Random Forest
   - Decision Tree
   - K-Nearest Neighbors (KNN)

### 🔹 Deep Learning Pipeline
- Custom **CNN architecture** implemented in **PyTorch**
- Trained end-to-end on raw pixel data
- Includes:
  - Batch normalization
  - Dropout regularization
  - Learning rate scheduling
- Evaluated using training/validation loss and accuracy curves

---
## 🗂 Dataset

- **ORL Face Dataset**
- 40 subjects
- 10 images per subject
- Controlled variations in lighting, facial expressions, and pose

---
