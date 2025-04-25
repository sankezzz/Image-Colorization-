import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Input: (B, 1, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 128, 16, 16)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # (B, 64, 32, 32)
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Load the full model directly (no need to define class again)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("color_model_full.pth", map_location=device)
model.eval()



# Streamlit UI
st.title("🎨 Colorize Black & White Image")

uploaded_file = st.file_uploader("Upload a grayscale image (32x32 recommended)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Grayscale Input", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 1, 32, 32)

    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()  # Shape: (3, 32, 32)

    color_image = transforms.ToPILImage()(output.clamp(0, 1))

    st.subheader("🎨 Colorized Output")
    st.image(color_image, use_column_width=True)

    st.subheader("📸 Side-by-Side Comparison")
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Grayscale")
    axs[1].imshow(color_image)
    axs[1].set_title("Colorized")
    for ax in axs: ax.axis("off")
    st.pyplot(fig)

