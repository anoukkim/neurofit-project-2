# MultiInputCNN Experiment

This repository contains code and results for an experiment involving a UNet model for segmentation and a MultiInputCNN model for classification. The goal of the project is to evaluate the performance of these models on a medical imaging dataset.

---

## Table of Contents
- [Dataset Information](#dataset-information)
- [Preprocessing](#preprocessing)
- [Model Details](#model-details)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Work](#future-work)

---

## Dataset Information

- **Dataset Source**: `lgg-mri-segmentation`
- **Data Split**:
  - Training Set: 1940
  - Validation Set: 416
  - Test Set: 416
- **Features**: 18 features (15 extracted + 3 tumor-related)
- **Target**: Binary classification (`Alive` or `Dead`)

---

## Preprocessing

1. **Image Resizing**: Images were resized to (256 x 256).
2. **Normalization**: Pixel values were normalized to `[0, 1]`.
3. **Tumor Features Extracted**:
   - Tumor Area
   - Mean Intensity
   - Standard Deviation of Intensity

---

## Model Details

### UNet (Segmentation)
- **Architecture**: Custom UNet for segmentation.
- **Pretrained**: Pretrained on Data.
- **Purpose**: Generate tumor masks from input images.

### MultiInputCNN (Classification)
- **Architecture**: Combines image and feature inputs for classification.
- **Purpose**: Classify patients as `Alive` or `Dead`.

---

## Training Details

- **Loss Function**: Binary Cross-Entropy Loss.
- **Optimizer**: Adam.
- **Learning Rate**: `1e-4`
- **Batch Size**: `32`
- **Number of Epochs**: `20`
- **Device**: CUDA (GPU)

---

## Evaluation Metrics

- **Confusion Matrix**:

  |                  | Predicted Alive | Predicted Dead |
  |------------------|-----------------|----------------|
  | **True Alive**   | 292             | 10             |
  | **True Dead**    | 36              | 78             |

- **Classification Report**:

```plaintext
              precision    recall  f1-score   support

       Alive       0.89      0.97      0.93       302
        Dead       0.89      0.68      0.77       114

    accuracy                           0.89       416
   macro avg       0.89      0.83      0.85       416
weighted avg       0.89      0.89      0.88       416
