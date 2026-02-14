# Technical Report

## Resources

* **GitHub Repo:** [hughmancoder/lettuce-dry-weight-prediction](https://github.com/hughmancoder/lettuce-dry-weight-prediction)
* **Competition Notes:** `documentation/Notes.md`

## Problem Statement

The goal is to predict **DryWeightShoot** (g/plant) using paired RGB and depth images.

* **Primary Metric:** Mean Absolute Error (MAE).
* **Dataset:**  labelled training samples.
* **Privileged Information:** Fresh weight and plant height are available during training to help regularise the model and improve feature representation.

---

## Strategy Implemented

### 1. Early-Fusion ResNet18 Architecture

Instead of a dual-stream approach, the model uses a single-stream **ResNet18** modified for **4-channel input** (RGB + Height Map).

* **Input Layer:** The first convolutional layer was expanded from 3 to 4 channels.
* **Weight Initialisation:** To leverage ImageNet pretraining, the first 3 channels use standard weights, while the 4th (height) channel is initialised using the mean of the RGB weights.
* **Backbone:** The final fully connected layer is replaced with an `Identity` block to extract 512-dimensional features.

### 2. Multi-Task Learning (MTL)

The model features three independent linear heads to predict primary and auxiliary targets simultaneously:

1. **Dry Weight** (Primary)
2. **Fresh Weight** (Auxiliary)
3. **Plant Height** (Auxiliary)

### 3. Weighted Multi-Task Loss

The model is optimised using a weighted Mean Absolute Error (MAE) loss function to prioritise the primary target while benefiting from the auxiliary signals:

---

## Preprocessing & Data Augmentation

### Preprocessing Pipeline

* **Height Mapping:** Raw depth data is transformed into a height map: .
* **Normalisation:** RGB channels are normalised using ImageNet statistics 

### Synchronised Augmentation

To maintain spatial alignment between the RGB image and the Height Map, the following transforms are applied **identically** to both inputs:

* **Geometric:** Random horizontal flips, random vertical flips, and random rotations
* **Photometric:** `ColorJitter` (brightness and contrast) is applied **only** to the RGB channels to prevent distorting the physical meaning of the height map.


## Training 

The training process uses a robust **5-fold Cross-Validation** strategy:

* **Optimiser:** `AdamW` with a learning rate of  and weight decay.
* **Validation:** The best model per fold is saved based on the validation MAE of the `DryWeightShoot` head.
* **Outputs:**
* Fold checkpoints: `weights/model/model_fold_*.pth`
* Out-of-fold (OOF) predictions: `outputs/model/oof_predictions.csv`
