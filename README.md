# Bone Fracture Classification using ViT + TCN

This project implements a hybrid deep learning model that combines a Vision Transformer (ViT) and a Temporal Convolutional Network (TCN) for binary classification of bone fractures in X-ray images.

The training is done using PyTorch and utilizes Google Colab with Google Drive integration for dataset management and model saving.

Developed web interface(Flask) using "web_app_flask.py" and "index.html" files on saved hybrid model. 


Project Overview:
1. Model Type: Hybrid Vision Transformer + TCN

2. Task: Binary classification (Fractured vs. Non-Fractured)

3. Dataset Format: Image folder structure

4. Platform: Google Colab / PyCharm

5. Framework: PyTorch + timm
   

Model Architecture
ViT (Vision Transformer):

1. Pretrained on ImageNet.

2. Used for feature extraction from 2D images.

3. Output: 768-dimensional feature vectors.

TCN (Temporal Convolutional Network):

1. Treats the extracted ViT features as temporal sequences.

2. 1D convolution layers process these features.

3. Global average pooling followed by a fully connected layer for classification.


Accuracy:
```
dataset1 on x-rays: 98.79%
dataset2 on x-rays: 99.83%
dataset3 on CT, MRI, X-rays: 80.20%
```

Datasets:
1. https://www.kaggle.com/code/prasadchaskar/bone-fracture-detection-97-accuracy-cnn/input

2. https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays

3. https://ribfrac.grand-challenge.org/


Dataset structure preferred:
```
Bone_Fracture_Binary_Classification/
│
├── train/
│   ├── Fractured/
│   └── Normal/
│
├── val/
│   ├── Fractured/
│   └── Normal/
│
└── test/
    ├── Fractured/
    └── Normal/
```

Web interface using Flask:

![image](https://github.com/user-attachments/assets/2a0867bb-a975-4b9f-851b-d75f39a519af)



![image](https://github.com/user-attachments/assets/f928a2cf-f4b4-4bfd-80a7-151ca565bbe2)


