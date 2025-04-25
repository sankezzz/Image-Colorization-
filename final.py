import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Define your model architecture
class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("color_model_full.pth", map_location=device)
model.eval()

# Streamlit UI
st.title("🎨 Colorize Black & White Image")

uploaded_file = st.file_uploader("Upload a grayscale image (32x32 recommended)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="🖤 Grayscale Input", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict color image
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()

    # Convert model output to image
    color_image = transforms.ToPILImage()(output.clamp(0, 1))
    sharpened_image = color_image.filter(ImageFilter.SHARPEN)

    # Display all three images side by side
    st.subheader("🖼️ Comparison: Grayscale | Colorized | Sharpened")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Grayscale", use_column_width=True)
    with col2:
        st.image(color_image, caption="Colorized", use_column_width=True)
    with col3:
        st.image(sharpened_image, caption="Sharpened", use_column_width=True)
