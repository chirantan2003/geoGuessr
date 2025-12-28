# Hierarchical Street-to-GPS Geolocation

A Computer Vision pipeline for precise geolocation estimation from multi-view street images in the USA. This project implements a hierarchical **"Scout & Sniper"** architecture, combining the semantic understanding of **StreetCLIP** with the structural feature extraction of **ConvNeXt V2**, refined by a custom **Residual Regressor**.

## Architecture Overview

The system operates in a coarse-to-fine manner to minimize geodesic error:

1.  **Phase 1: The Scout (Classification)**
    * **Models:** Ensemble of `StreetCLIP (ViT-Large)` and `ConvNeXt V2 (Large)`.
    * **Input:** 4 Cardinal Views (North, East, South, West) per location.
    * **Mechanism:** Multi-View Transformer Fusion to create a holistic scene descriptor.
    * **Output:** Predicts the coarse state/region (Softmax probability distribution).

2.  **Phase 2: The Sniper (Residual Regression)**
    * **Model:** Residual MLP Regressor attached to the frozen backbone.
    * **Input:** Fused Visual features + Top-1 State Embedding + Confidence Score.
    * **Output:** Predicts the precise **(Latitude, Longitude) offset** from the predicted state's centroid.

---

## 🚀 Key Features

* **Multi-View Fusion:** Uses a Transformer Encoder to attend across N/E/S/W views, handling missing angles and synthesizing 360° context.
* **Logit-Level Ensemble:** Fuses predictions from CLIP (Semantic expert) and ConvNeXt (Texture expert) in the logit space for robust classification.
* **Residual Learning:** Instead of regressing global coordinates directly (which is unstable), the model learns local offsets, significantly reducing distance error.
* **Haversine Loss:** Optimized directly on geodesic distance (km) rather than Euclidean space.

---
