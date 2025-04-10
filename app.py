import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat or Dog Classifier")

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
model.load_state_dict(torch.load("vit_cats_dogs.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = "Cat" if predicted.item() == 0 else "Dog"

    st.markdown(f"### 🧠 Prediction: **{label}**")

