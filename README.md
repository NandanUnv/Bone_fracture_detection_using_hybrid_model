🦴 Bone Fracture Classification using ViT + TCN
This project implements a hybrid deep learning model that combines a Vision Transformer (ViT) and a Temporal Convolutional Network (TCN) for binary classification of bone fractures in X-ray images.

The training is done using PyTorch and utilizes Google Colab with Google Drive integration for dataset management and model saving.


📌 Project Overview:
1. Model Type: Hybrid Vision Transformer + TCN

2. Task: Binary classification (Fractured vs. Non-Fractured)

3. Dataset Format: Image folder structure

4. Platform: Google Colab

5. Framework: PyTorch + timm
   

🧠 Model Architecture
ViT (Vision Transformer):

Pretrained on ImageNet.

Used for feature extraction from 2D images.

Output: 768-dimensional feature vectors.

TCN (Temporal Convolutional Network):

Treats the extracted ViT features as temporal sequences.

1D convolution layers process these features.

Global average pooling followed by a fully connected layer for classification.

