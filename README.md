# Salient Object Detection (SOD) from Scratch

This project implements a Salient Object Detection (SOD) system from scratch using PyTorch.  
The goal is to detect and segment the most visually important object in an image and generate a saliency mask with an overlay visualization.

---

## Project Overview

The project includes a complete deep learning pipeline:

- Dataset loading and preprocessing
- Image resizing and normalization
- Train, validation, and test split
- Data augmentation
- CNN encoder-decoder baseline model
- Improved model variants
- U-Net style final model with skip connections
- Training and validation loop
- Evaluation metrics
- Visualization of predictions
- Jupyter Notebook demo
- Checkpoint saving and resume functionality

---

## Dataset

Dataset used: **ECSSD**

Dataset source:  
https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html#ref_2

The dataset contains 1000 images with corresponding saliency masks.

Expected folder structure:

```text
data/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

The dataset is split into:

- 70% training
- 15% validation
- 15% testing

---

---

## Model Improvements

Several experimental configurations were implemented and compared:

| Configuration | Description |
|---|---|
| Baseline CNN | Simple encoder-decoder CNN |
| BatchNorm CNN | Baseline model with Batch Normalization |
| BatchNorm + Dropout | Added Dropout regularization |
| Stronger Augmentation + Lower LR | Added stronger augmentation and reduced learning rate |
| U-Net CNN | Final model with skip connections |
| U-Net 224x224 | Final best model using higher input resolution |

---

## Final Results

The best result was achieved with the U-Net style model using 224x224 input resolution.

| Model | IoU | Precision | Recall | F1-score | MAE |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.3763 | 0.5423 | 0.5600 | 0.5429 | 0.2780 |
| BatchNorm CNN | 0.3771 | 0.6369 | 0.4848 | 0.5435 | 0.2326 |
| BatchNorm + Dropout | 0.4371 | 0.5728 | 0.6564 | 0.6058 | 0.2606 |
| Stronger Augmentation + Lower LR | 0.4157 | 0.5731 | 0.6106 | 0.5841 | 0.2690 |
| U-Net 128x128 | 0.5406 | 0.7612 | 0.6558 | 0.6980 | 0.1810 |
| U-Net 224x224 Final | 0.5674 | 0.7401 | 0.7112 | 0.7199 | 0.1717 |

---

---

## How to Run

### 1. Prepare dataset

Place the ECSSD images and masks into:

```text
data/images/
data/masks/
```

### 2. Train the model

```bash
python train.py
```

The best model is saved in:

```text
saved_models/unet_model.pth
```

A checkpoint is also saved in:

```text
checkpoints/unet_checkpoint.pth
```

If training is interrupted, it automatically resumes from the last checkpoint.

### 3. Evaluate the model

```bash
python evaluate.py
```

This prints the test metrics and saves visual examples in:

```text
outputs/unet/
```

### 4. Run the demo

Open:

```text
demo_notebook.ipynb
```

The notebook shows:

- Input image
- Ground-truth mask
- Predicted saliency mask
- Overlay visualization
- Inference time per image

---

## Evaluation Metrics

The model is evaluated using:

- Intersection over Union (IoU)
- Precision
- Recall
- F1-score
- Mean Absolute Error (MAE)

---

## Final Model

The final selected model is:

```text
UNetSODModel
```

Saved model:

```text
saved_models/unet_model.pth
```

This model is used for the final demo and report.

---

## Notes

The full dataset is not included in this repository.  
Please download ECSSD from the official dataset page and place the files into the correct folders before training or evaluation.
