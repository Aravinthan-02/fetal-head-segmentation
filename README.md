# Fetal Head Segmentation using U-Net

A deep learning project for automatic fetal head segmentation from 2D ultrasound images using a lightweight **U-Net** architecture implemented in **PyTorch**.

## Overview

The model takes a fetal ultrasound image as input and predicts a binary mask representing the fetal head region.

The project uses the **HC18 (Head Circumference Challenge)** dataset containing fetal ultrasound images and corresponding head annotations.

### Pipeline

Ultrasound Image → Preprocessing → U-Net → Sigmoid → Binary Segmentation Mask

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pillow
- CUDA
- NVIDIA GPU

## Model

A lightweight U-Net architecture consisting of:

- Encoder with convolution and ReLU layers
- Max pooling for downsampling
- Bottleneck feature representation
- Decoder with transposed convolutions
- Skip connections for spatial information preservation
- 1×1 convolution for binary segmentation output

## Loss Function

The model is trained using:

**BCE + Dice Loss**

- **Binary Cross Entropy (BCE):** Pixel-level classification loss
- **Dice Loss:** Encourages overlap between predicted and ground-truth masks

Optimization is performed using the **Adam optimizer**.

## Dataset Split

- Training: 70% — 699 images
- Validation: 15% — 149 images
- Testing: 15% — 151 images

## Results

| Metric | Test Result |
|---|---:|
| Test Loss | 0.5037 |
| Dice Score | 0.7981 |
| IoU | 0.6874 |

