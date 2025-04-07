
import streamlit as st
import torch
import numpy as np
from PIL import Image
from models.unet import UNet
import matplotlib.pyplot as plt
import os
import json

st.set_page_config(page_title="Circle Segmentation", layout="centered")
st.title("🟢 Circle Segmentation Demo")
st.write("Upload a grayscale image to segment the circles and view training results.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load("src/model_weights.pth", map_location=device))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    img_np = np.array(image).astype(np.float32) / 255.0
    x = torch.tensor(img_np[None, None, ...], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(x)
        mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    st.subheader("🖼 Results")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    with col2:
        st.image(mask * 255, caption="Predicted Mask", use_container_width=True)

# --- TRAINING CURVES ---
if os.path.exists("src/train_log.json"):
    st.subheader("Loss and IoU Curves")

    with open("src/train_log.json", "r") as f:
        log = json.load(f)

    epochs = list(range(len(log["train"])))

    fig, ax = plt.subplots()
    ax.plot(epochs, log["train"], label="Train Loss")
    ax.plot(epochs, log["val"], label="Val Loss")
    if "test" in log:
        ax.plot(epochs, log["test"], label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    if "val_iou" in log and "test_iou" in log:
        fig2, ax2 = plt.subplots()
        ax2.plot(epochs, log["val_iou"], label="Val IoU")
        ax2.plot(epochs, log["test_iou"], label="Test IoU")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

# --- EXAMPLES BELOW CURVES ---
if os.path.exists("src/val_examples.npy"):
    st.subheader("🔍 Sample Predictions")
    data = np.load("src/val_examples.npy", allow_pickle=True).item()
    imgs, labels, preds = data["images"], data["labels"], data["preds"]

    for i in range(min(5, len(imgs))):
        st.markdown(f"**Example {i+1}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(imgs[i][0].squeeze(), caption="Input", use_container_width=True)
        with col2:
            label_img = labels[i].squeeze()
            label_slice = label_img[1] if label_img.ndim == 3 else label_img
            st.image(label_slice, caption="Ground Truth", use_container_width=True)
        with col3:
            pred_mask = (preds[i][0].squeeze() > 0.5).astype(np.uint8) * 255
            st.image(pred_mask, caption="Prediction", use_container_width=True)
