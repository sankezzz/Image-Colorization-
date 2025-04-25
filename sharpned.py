import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn


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


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("color_model_full.pth", map_location=device)
model.eval()


# Streamlit UI
st.title("🎨 Grayscale to Color + Sharpener")

uploaded_file = st.file_uploader("Upload a grayscale image (32x32 preferred)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="🖤 Grayscale Input", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()

    color_image = transforms.ToPILImage()(output.clamp(0, 1))

    # ✅ Sharpen using custom kernel (no smoothing)
    image_cv = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2BGR)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_cv = cv2.filter2D(image_cv, -1, sharpen_kernel)
    sharpened_pil = Image.fromarray(cv2.cvtColor(sharpened_cv, cv2.COLOR_BGR2RGB))

    # Optional small contrast boost (very mild)
    sharpened_final = ImageEnhance.Contrast(sharpened_pil).enhance(1.05)

    # 📸 Side-by-side display
    st.subheader("🖼️ Side-by-Side Comparison")
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Grayscale")

    axs[1].imshow(color_image)
    axs[1].set_title("Colorized")

    axs[2].imshow(sharpened_final)
    axs[2].set_title("Sharpened")

    for ax in axs:
        ax.axis("off")

    st.pyplot(fig)
