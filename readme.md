# Hierarchical Street-to-GPS Geolocation

A Computer Vision pipeline for precise geolocation estimation from multi-view street images in the USA. This project implements a hierarchical **"Scout & Sniper"** architecture, combining the semantic understanding of **StreetCLIP** with the structural feature extraction of **ConvNeXt V2**, refined by a custom **Residual Regressor**.

## Performance

| Metric | Accuracy |
| :--- | :--- |
| **Validation Accuracy** | **94.86%** |
| **Kaggle Private Score** | **94.70%** |

## Architecture Overview

The system operates in a coarse-to-fine manner to minimize geodesic error:

1.  **Phase 1 & 2: The Scout (Classification)**
    * **Models:** Ensemble of `StreetCLIP (ViT-Large)` and `ConvNeXt V2 (Large)`.
    * **Input:** 4 Cardinal Views (North, East, South, West) per location.
    * **Mechanism:** Multi-View Transformer Fusion to create a holistic scene descriptor.
    * **Output:** Predicts the coarse state/region (Softmax probability distribution).

2.  **Phase 3: The Sniper (Residual Regression)**
    * **Model:** Residual MLP Regressor attached to the frozen backbone.
    * **Input:** Fused Visual features + Top-1 State Embedding + Confidence Score.
    * **Output:** Predicts the precise **(Latitude, Longitude) offset** from the predicted state's centroid.

---

## Key Features

* **Multi-View Fusion:** Uses a Transformer Encoder to attend across N/E/S/W views, handling missing angles and synthesizing 360° context.
* **Logit-Level Ensemble:** Fuses predictions from CLIP (Semantic expert) and ConvNeXt (Texture expert) in the logit space for robust classification.
* **Residual Learning:** Instead of regressing global coordinates directly (which is unstable), the model learns local offsets, significantly reducing distance error.
* **Haversine Loss:** Optimized directly on geodesic distance (km) rather than Euclidean space.

---

# Execution Order

To reproduce the results, run the scripts in this specific sequence. This pipeline first trains the individual classifiers, then refines the location using the regressor, and finally combines them.

### 1. Train StreetCLIP (Classifier)

```bash
python classifier_phase_1.py
```

* **Output:** `checkpoints_streetclip_classifier/best_model_cls.pth`

### 2. Train ConvNeXt (Classifier)

```bash
python classifier_phase_2.py
```

* **Output:** `checkpoints_convnext/best_convnext_cls.pth`

### 3. Train Residual Regressor (Regressor)

```bash
python regressor_phase_3.py
```

* **Output:** `checkpoints_streetclip_classifier/best_residual.pth`

### 4. Run Ensemble Inference

```bash
python ensemble_final.py
```

* **Output:** `submission_ensemble_final_logitsv2.csv`

---