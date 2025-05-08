
# Hybrid model code


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from google.colab import drive
from PIL import Image

# Mount Google Drive
drive.mount('/content/drive')

# Data Directory (Set your dataset path)
TRAIN_DIR = '/train'
VAL_DIR = '/val'
TEST_DIR = '/test'

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super(CustomImageFolder, self).__getitem__(index)
            except (OSError, IOError):
                print(f"Skipping corrupted image at index {index}")
                index = (index + 1) % len(self.samples)

# Load Datasets
train_dataset = CustomImageFolder(root=TRAIN_DIR, transform=transform)
val_dataset = CustomImageFolder(root=VAL_DIR, transform=transform)
test_dataset = CustomImageFolder(root=TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Vision Transformer Feature Extractor
class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

    def forward(self, x):
        return self.vit(x)

# Temporal Convolutional Network
class TCN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=-1)
        return self.fc(x)

# Hybrid Model Combining ViT + TCN
class ViT_TCN(nn.Module):
    def __init__(self, num_classes=2):
        super(ViT_TCN, self).__init__()
        self.vit = ViTFeatureExtractor()
        self.tcn = TCN(input_dim=768, num_classes=num_classes)

    def forward(self, x):
        features = self.vit(x)
        features = features.unsqueeze(2)
        return self.tcn(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_TCN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save Model to Google Drive
model_path = 'name_your_model'
torch.save(model.state_dict(), model_path)
print(f'Model saved at {model_path}')




#testing model
# Load Model for Testing
model.load_state_dict(torch.load('path_of_saved_model'))
model.eval()

# Testing Loop
def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Run Testing
test_model(model, test_loader)
