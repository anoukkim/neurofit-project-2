# UNet Pretraining Experiment

This experiment focuses on pretraining a UNet model for medical image segmentation using the LGG MRI Segmentation dataset. The goal is to evaluate the model's segmentation performance based on Dice scores and loss values.

---

## Dataset Information

- **Dataset Source**: `lgg-mri-segmentation`
- **Data Split**:
  - **Training Set**: 1940 samples (70%)
  - **Validation Set**: 416 samples (15%)
  - **Test Set**: 416 samples (15%)
- **Features**:
  - **Images**: Grayscale MRI scans resized to (256 x 256)
  - **Masks**: Corresponding binary tumor masks resized to (256 x 256)

---

## Preprocessing

1. **Image Transformation**:
   - **Resize**: Images and masks resized to (256 x 256)
   - **Normalization**: Pixel values normalized to the range `[0, 1]`
2. **Train-Validation-Test Split**:
   - **Training**: 70% of the dataset
   - **Validation**: 15% of the dataset
   - **Testing**: 15% of the dataset

---

## Model Details

- **Architecture**: UNet with the following components:
  - **Encoder**: 4 convolutional blocks with downsampling
  - **Bottleneck**: 1 convolutional block
  - **Decoder**: 4 upsampling blocks with skip connections
  - **Final Layer**: 1x1 convolution with sigmoid activation for binary segmentation
- **Input**: Single-channel grayscale images (1x256x256)
- **Output**: Single-channel binary masks (1x256x256)

---

## Training Details

- **Loss Function**: Binary Cross-Entropy Loss (BCELoss)
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 32
- **Number of Epochs**: 50
- **Metrics**:
  - **Loss**: Binary Cross-Entropy Loss
  - **Dice Score**: Evaluates overlap between predicted and ground truth masks
- **Device**: CUDA (GPU) or CPU

---

## Results

### Final Performance Metrics (Test Set)

| Metric      | Value  |
|-------------|--------|
| Test Loss   | 0.0097 |
| Test Dice   | 0.7411 |

---

## Visualization Results

### Example 1
**Original Image**:
![Original Image](images/example1_original.png)

**Ground Truth Mask**:
![Ground Truth Mask](images/example1_ground_truth.png)

**Predicted Mask**:
![Predicted Mask](images/example1_predicted.png)

---

## Summary

- The UNet model was successfully trained for medical image segmentation.
- **Best Validation Dice Score**: 0.7411  
- **Test Dice Score**: 0.7411
- The results demonstrate that the model can effectively identify and segment tumor regions.
