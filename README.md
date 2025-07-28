# PneumoniaMNIST Classifier

A PyTorch-based tool for loading, analyzing, and evaluating the [PneumoniaMNIST](https://medmnist.com/) dataset. This project includes data loading, preprocessing, visualization of class distribution, and performance evaluation using common classification metrics.

## About the Project

This project aims to classify pneumonia cases using chest X-ray images from the PneumoniaMNIST dataset. It leverages PyTorch and Scikit-learn to provide a simple, educational framework for working with medical image classification tasks.

## Dataset

- **Source**: [MedMNIST – PneumoniaMNIST](https://medmnist.com/)
- **Format**: 28x28 grayscale chest X-ray images
- **Classes**:
  - 0: Normal
  - 1: Pneumonia

The dataset is automatically downloaded via the `medmnist` Python package.

## Features

- Automatic dataset download and preprocessing
- Class distribution visualization
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - ROC Curve & AUC
