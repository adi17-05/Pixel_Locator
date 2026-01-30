# Pixel Localization (50×50) using Deep Learning (Coordinate Regression)

## Overview
This project solves a synthetic computer vision regression task:
**Given a 50×50 grayscale image where exactly one pixel = 255 and all others are 0, predict the (x, y) coordinates of the bright pixel.**

The solution is implemented in TensorFlow/Keras and demonstrates multiple modeling strategies. The approach and reasoning (dataset design + model framing + evaluation) are emphasized more than raw accuracy, matching the assignment evaluation criteria.

## Problem Statement

### Input
- A grayscale image of shape (50, 50)
- Exactly one bright pixel has intensity 255
- All other pixels are 0

### Output
- The coordinates **(x, y)** of the bright pixel
- `x` = column index (0–49)
- `y` = row index (0–49)

### Goal
Train a deep learning model to predict the pixel location from the image.

### Why This Is a Regression Task (Not Classification)
Even though the output is a location, the prediction is naturally modeled as continuous coordinates (especially when we want smooth learning and stable optimization).
- **Input:** image tensor (50×50×1)
- **Output:** 2 real values (x, y)
- **Loss:** Mean Squared Error (MSE) / MAE on coordinates

This keeps the pipeline simple, interpretable, and aligned with coordinate prediction tasks in vision.

## Dataset Generation (Synthetic)
There is no standard dataset matching this exact constraint, so the dataset is generated programmatically.

### Dataset properties
- Each sample contains exactly one bright pixel
- Pixel position is sampled **uniformly at random** across all 2,500 locations
- This prevents spatial bias and ensures full coverage

### Why synthetic is the right choice here
- Full control over constraints
- Balanced coordinate distribution
- Fast generation without manual labeling
- Common practice for controlled learning tasks

## Modeling Strategy (3 Approaches)
To show “approach is more important than accuracy”, the notebook implements three models (progressively more spatially aware):

### 1) MLP Baseline (Flatten → Dense → (x, y))
- Treats the image as a flat vector
- Good baseline to verify the pipeline end-to-end
- **Saved as:** `best_mlp.keras`

### 2) CNN Coordinate Regressor (Conv layers → Dense → (x, y))
- Preserves spatial locality via convolutions
- Learns positional structure more naturally than MLP
- **Saved as:** `best_cnn.keras`

### 3) Heatmap + Soft-Argmax / Coordinate Head (Spatial probability → (x, y))
- Predicts a heatmap-like representation and converts it to coordinates
- Often more stable/interpretable for localization tasks
- **Saved as:** `best_m3_coords.keras`

The final selected model can also be exported to TensorFlow Lite for lightweight inference:
- **Exported as:** `model.tflite`

## Repository Structure
```
.
├── Pixel_Prediction.ipynb          # Main notebook: training + evaluation + plots
├── Dataset generation.ipynb        # Dataset creation notebook
├── Pixel_dataset/                  # Generated dataset (TFRecords)
├── logs/                           # TensorBoard logs
├── checkpoints/                    # Training checkpoints
├── best_mlp.keras                  # Best saved MLP model
├── best_cnn.keras                  # Best saved CNN model
├── best_m3_coords.keras            # Best saved Model-3 coordinate model
├── model.tflite                    # TFLite export (optional)
├── README.md                       # Documentation
└── requirements.txt                # Dependencies list
```

## Setup & Installation

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
Running the following command to install the required packages:
```bash
pip install -r requirements.txt
```

### 3) Run the notebook
Launch Jupyter:
```bash
jupyter notebook
```
Open: `Pixel_Prediction.ipynb`

## How to Run (Typical Flow)

1.  **Generate dataset:**
    - Run the cells in `Pixel_Prediction.ipynb` (Steps 1-3) OR run the `Dataset generation.ipynb` notebook.
2.  **Train models:**
    - The notebook trains all 3 models and tracks metrics.
3.  **Evaluate:**
    - The notebook shows:
        - Training/validation loss curves
        - Predicted vs ground truth coordinate plots
        - Qualitative overlays of predicted point vs true point
4.  **Save best models:**
    - Best models are automatically saved as `.keras` files.
5.  **Export to TFLite (Optional):**
    - Produces `model.tflite`.

## Training Logs & Visualization
TensorBoard logs are stored in `logs/`. You can view them via:
```bash
tensorboard --logdir logs
```

## Evaluation Metrics (What We Report)
The notebook typically reports:
- **MSE / MAE** for coordinate regression
- **Mean Euclidean Distance Error** in pixel space (optional but very interpretable)

### Visual Checks Included
For a set of samples, the notebook visualizes:
- The input image
- The true bright pixel position
- The predicted (x, y) position

This confirms the model is correctly learning localization, not just minimizing loss blindly.

## Notes on Reproducibility
- Use a fixed random seed in dataset generation and training (where applicable)
- Uniform sampling ensures consistent coordinate coverage
- Saving best checkpoints ensures results are reproducible across runs

## Submission Checklist
- [x] Deep learning solution that predicts (x, y)
- [x] Synthetic dataset generation + rationale
- [x] Notebook includes training logs and graphs
- [x] Ground-truth vs predicted coordinate visualization
- [x] Code is organized and readable
- [x] Dependencies provided (requirements.txt)
- [x] Saved models included (.keras, .tflite)

## Author
**Name:** ADITYA D<br>
**Email:** adi.divakar1705@gmail.com<br>
**Phone:** 7338211060<br>
**University / Course:** SJCE-Mysore/CSE
