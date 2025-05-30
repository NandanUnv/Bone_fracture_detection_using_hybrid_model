import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import torch.nn.functional as F
from PIL import Image
import numpy as np
from flask import Flask, render_template, request


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


def model_pred(f):
    model = ViT_TCN()  # Instantiate your model architecture
    model.load_state_dict(torch.load('C:/Users/nanda/PycharmProjects/FlaskProject/Hybrid_model_vit_tcn.pth',
                                     map_location=torch.device('cpu')))  # Load saved weights
    model.eval()  # Set the model to evaluation mode

    # Image preprocessing (example for ViT and CNN hybrid model)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # If your images are grayscale, use single value for mean and std
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
    ])

    # Load your image
    image = Image.open(f)

    # Preprocess the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_batch = input_batch.to(device)
    model.to(device)  # Load the model to GPU if available

    # Perform the forward pass and get the predictions
    with torch.no_grad():  # No need to compute gradients during inference
        output = model(input_batch)

    # If it's a classification model, use softmax to get probabilities
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(output)

    # Get the class with the highest probability
    predicted_class = torch.argmax(probabilities, dim=1)

    # Print or visualize the result
    print(f'Predicted class: {predicted_class.item()}')
    return predicted_class.item()


app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
      f = request.files['img']
      model_prediction = model_pred(f)
      return render_template('index.html', prediction = model_prediction)


if __name__ == '__main__':
  app.run(debug=True)
