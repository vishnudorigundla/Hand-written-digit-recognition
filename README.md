# Handwritten Digit Recognition using Optimized K-Nearest Neighbors
###  Project Description

This project implements a handwritten digit recognition system using the MNIST dataset and an optimized K-Nearest Neighbors (KNN) algorithm.

The goal of the project is to:

Achieve high classification accuracy (~95%) on MNIST.

Reduce prediction latency by 25% through vectorized NumPy operations.

Perform hyperparameter tuning using GridSearchCV and K-fold cross-validation.

Build a scalable machine learning pipeline that processes large image datasets 30% faster.

The project demonstrates how classical machine learning models can be optimized for real-time applications like banking check digit recognition, postal code reading, and OCR-based document digitization.

### Features

✅ Achieved 95% accuracy on MNIST dataset.

⚡ Optimized prediction with vectorized NumPy operations.

🔍 GridSearchCV + K-fold cross-validation for hyperparameter tuning.

📊 Scalable pipeline with improved processing efficiency.

📈 Visualizations of misclassifications, confusion matrix, and performance comparisons.

📂 Dataset

MNIST Handwritten Digit Dataset

70,000 grayscale images (28×28 pixels).

Source: OpenML MNIST

###  Tech Stack

Programming Language: Python

Libraries:

scikit-learn → KNN, GridSearchCV, pipelines

NumPy → vectorized operations

Matplotlib, Seaborn → visualizations

JupyterLab → development environment

### Project Workflow

Data Loading & Exploration

Load MNIST dataset from OpenML

Visualize sample digits

Data Preprocessing

Normalize pixel values (0–1)

Flatten images into 784 features

Baseline KNN Model

Train & test with default parameters

Evaluate initial accuracy & latency

### Optimization

GridSearchCV for best k, distance metric, and weights

K-fold cross-validation to reduce overfitting

NumPy vectorization for faster distance computation

Pipeline & Final Model

Build scalable ML pipeline

Evaluate on test set

Generate confusion matrix & classification report

Performance Analysis

Compare baseline vs optimized accuracy

Show latency improvements

### Results

Accuracy: ~95%

Prediction Latency: Reduced by ~25%

Pipeline Efficiency: 30% faster on large datasets

### Real-World Applications

Banking: Automatic check digit recognition.

Postal Services: ZIP/postal code recognition.

OCR Systems: Digit extraction from scanned forms.

Education: Handwriting recognition learning apps.

Mobile Input: Handwriting-to-text applications.
