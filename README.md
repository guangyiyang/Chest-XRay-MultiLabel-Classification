# Chest X-Ray Multi-Label Classification

A deep learning project for multi-label classification of thoracic conditions from chest X-ray images.

## Overview

This project builds and evaluates a deep learning model using chest X-ray images to identify multiple thoracic conditions.

The model is trained to recognize the following labels:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Pleural
- No Findings

Since a single chest X-ray may contain more than one condition, this task is treated as a **multi-label image classification problem**.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Model

This project uses a pretrained **DenseNet121** model from PyTorch's `torchvision` library for multi-label chest X-ray classification.

The model is initialized with **ImageNet pretrained weights** and adapted using transfer learning. The original classification layer is replaced with a custom output layer corresponding to the target labels.

Dropout is optionally applied before the final linear layer to reduce overfitting.

The model is trained using **AdamW** as the optimizer and **BCEWithLogitsLoss** as the loss function for multi-label classification.
