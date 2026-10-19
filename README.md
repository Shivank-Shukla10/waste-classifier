# ♻️ WasteVision AI — Explainable Waste Classification System

An end-to-end deep learning application for classifying waste images into 12 categories using **MobileNetV2 Transfer Learning**.

The project includes model training, evaluation, explainability with Grad-CAM, MLflow experiment tracking, FastAPI inference API, Docker deployment, GitHub Actions CI, and a Streamlit web interface.

---

## 🚀 Project Overview

Waste classification is an important computer vision problem for automated waste segregation and recycling systems.

This project builds a complete production-oriented pipeline:

```text
Waste Image
     ↓
Streamlit UI
     ↓
FastAPI REST API
     ↓
MobileNetV2
     ↓
Waste Classification
     ↓
Top-3 Predictions
     ↓
Grad-CAM Explainability
```
