# Wind Turbine Classifier from Satellite Imagery

A custom Convolutional Neural Network (CNN) built with PyTorch to classify the presence of wind turbines in satellite image patches.


## Overview

This project tackles binary image classification on satellite patches: does the image contain a wind turbine, or not?

The model is a CNN trained from scratch. The goal was to understand the full pipeline while keeping the architecture simple and interpretable.



## Model Architecture


Input (3 × 128 × 128)
|
Conv2d(3 --> 16, 3×3) + ReLU + MaxPool --> (16 × 64 × 64)
|
Conv2d(16 --> 8, 3×3) + ReLU + MaxPool --> (8 × 32 × 32)
|
Flatten --> 8192
|
Linear(8192 --> 32) + ReLU
|
Linear(32 --> 2)



## Dataset

Airbus Wind Turbines Patches — available on Kaggle:  
[https://www.kaggle.com/datasets/airbusgeo/airbus-wind-turbines-patches]



## Training Details

| Parameter | Value |
---------------------
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | CrossEntropyLoss |
| Epochs | 20 |
| Batch size | 32 |
| Input size | 128 × 128 |

  Data augmentation (train only):
- Random horizontal and vertical flip
- Random rotation (±90°)
- Color jitter (brightness ±0.2)



## Results

Validation accuracy: 94.8%


## Requirements

torch
torchvision

Install with:

bash
pip install torch torchvision



## Usage

1. Download the dataset from Kaggle and organize it as:

Dataset/
|
|--Train/
|  |--Turbine/
|  |--No_Turbine/
|
|--Validation/
   |--Turbine/
   |--No_Turbine/

2. Update the `train_path` and `val_path` variables in the script to point to your local paths 

3. Run:

bash
python wind_turbine_classifier.py


## Notes

This project was developed independently, exploring deep learning applied to satellite imagery and remote sensing.


*Developed by Samuele Gentile — 2026*
