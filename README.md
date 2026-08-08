# Chest-XRay-MultiLabel-Classification

A deep learning project for classifying multiple thoracic diseases from chest X-ray images.

## Overview

This project builds and evaluates a deep learning model using chest X-ray images to identify multiple categories of thoracic diseases.

The model is trained to recognize the following diagnoses:

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

Since a single chest X-ray may contain more than one diagnosis, this task is treated as a **multi-label image classification problem**.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Model

A Convolutional Neural Network (CNN) is used to extract visual features from chest X-ray images and predict multiple disease labels.

The output layer uses a **sigmoid activation function**, allowing the model to independently predict the probability of each diagnosis.
