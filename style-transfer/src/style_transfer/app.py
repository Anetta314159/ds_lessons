import streamlit as st
import torch
from PIL import Image
import io
import os
import requests
import torch.nn.functional as F
from torchvision import transforms

from style_transfer.utils import image_loader
from style_transfer.model import gram_matrix, get_vgg

# Streamlit UI settings
st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title("Neural Style Transfer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(x, model, layers):
    """
    Извлекает признаки с заданных сверточных слоев VGG.

    Args:
        x (torch.Tensor): Входной тензор изображения.
        model (nn.Sequential): Сверточная часть VGG-сети.
        layers (list): Список имён слоев, признаки которых нужно извлечь.

    Returns:
        dict: Словарь {имя_слоя: тензор_признаков}.
    """
    feats = {}
    i = 0
    for layer in model:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            name = f'conv_{i}'
            if name in layers:
                feats[name] = x
    return feats

# === SIDEBAR UI ===
st.sidebar.header("Parameters")
transfer_mode = st.sidebar.radio("Transfer mode", ["Fast (pretrained)", "Gatys-style (slow)"], key="transfer_mode_radio")

# Gatys-style parameters
if transfer_mode == "Gatys-style (slow)":
    vgg_choice = st.sidebar.selectbox("VGG Backbone", ["vgg19", "vgg16"], key="vgg_choice")
    num_steps = st.sidebar.slider("Steps", 50, 1000, 300, step=50, key="num_steps")
    style_weight = st.sidebar.slider("Style weight (×10⁶)", 1, 10, 1, key="style_weight") * 1e6
    content_weight = st.sidebar.slider("Content weight", 1, 10, 1, key="content_weight")
else:
    # Fast Transfer — выбор модели
    fast_model_dir = "fast_models/"
    os.makedirs(fast_model_dir, exist_ok=True)
    available_models = [f for f in os.listdir(fast_model_dir) if f.endswith(".pth")]
    if not available_models:
        st.sidebar.warning("No fast models found in fast_models.")
        selected_model = None
    else:
        selected_model = st.sidebar.selectbox("Choose Fast Style Model", available_models)

# === FILE UPLOADS ===
content_file = st.file_uploader("Upload content image", type=["jpg", "png", "jpeg"], key="content_upload")
style_file_disabled = (transfer_mode == "Fast (pretrained)")
style_file = st.file_uploader("Upload style image", type=["jpg", "png", "jpeg"], disabled=style_file_disabled, key="style_upload")

# Display input images
if content_file:
    content_img = Image.open(content_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(content_img, caption="Content", use_container_width=True)
    with col2:
        if transfer_mode == "Gatys-style (slow)" and style_file:
            style_img = Image.open(style_file).convert("RGB")
            st.image(style_img, caption="Style", use_container_width=True)

# === INFERENCE ===
if st.button("Run Style Transfer", key="run_button") and content_file:
    with st.spinner("Processing..."):
        content_file.seek(0)
        content_tensor = image_loader(content_file, device)
        content_file.seek(0)

        if transfer_mode == "Gatys-style (slow)":
            # Отправка на сервер Gatys-style
            if not style_file:
                st.error("Style image is required for Gatys-style transfer.")
                st.stop()
            style_file.seek(0)
            files = {
                "content": content_file,
                "style": style_file,
            }
            params = {
                "steps": num_steps,
                "style_weight": style_weight,
                "content_weight": content_weight,
            }
            response = requests.post("http://127.0.0.1:8000/train-gatys", files=files, params=params)
        else:
            # Отправка Fast Style Transfer
            if not selected_model:
                st.error("No fast model selected.")
                st.stop()
            temp_path = "temp_content.jpg"
            with open(temp_path, "wb") as f:
                f.write(content_file.read())
            with open(temp_path, "rb") as img_file:
                response = requests.post(
                    "http://127.0.0.1:8000/stylize",
                    files={"content": img_file},
                    params={"model_name": selected_model}
                )

        if response.status_code != 200:
            st.error(f"FastAPI error: {response.status_code} - {response.text}")
            st.stop()

        # === Обработка результата ===
        # Преобразуем результат в PIL и в тензор для метрик
        image_bytes = response.content
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        transform_post = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        output_tensor = transform_post(image_pil).unsqueeze(0).to(device)

        # Display result and metrics
        st.success("✨ Stylization complete!")
        result_col, metrics_col = st.columns([2, 1])

        with result_col:
            st.image(image_pil, caption="🎨 Stylized Result", use_container_width=True)

        with metrics_col:
            st.subheader("Metrics")
            vgg = get_vgg("vgg19").to(device).eval()
            layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

            with torch.no_grad():
                output_f = extract_features(output_tensor, vgg, layers)
                content_f = extract_features(content_tensor, vgg, layers)

                # Content Loss
                content_loss = F.mse_loss(output_f['conv_4'], content_f['conv_4']).item()
                st.markdown(f"- **Content Loss**: `{content_loss:.4f}`")

                # Style Loss
                if transfer_mode == "Gatys-style (slow)":
                    style_tensor = image_loader(style_file, device)
                    style_features = extract_features(style_tensor, vgg, layers)
                    style_grams = {l: gram_matrix(style_features[l]) for l in layers}
                    style_loss = sum(
                        F.mse_loss(gram_matrix(output_f[l]), style_grams[l]) for l in layers
                    ).item()
                    st.markdown(f"- **Style Loss**: `{style_loss:.4f}`")
                else:
                    fallback = "data/style.jpg"
                    if style_file:
                        style_tensor = image_loader(style_file, device)
                    elif os.path.exists(fallback):
                        style_tensor = image_loader(fallback, device)
                    else:
                        style_tensor = None

                    if style_tensor is not None:
                        style_f = extract_features(style_tensor, vgg, layers)
                        style_grams = {l: gram_matrix(style_f[l]) for l in layers}
                        output_grams = {l: gram_matrix(output_f[l]) for l in layers}
                        style_loss = sum(
                            F.mse_loss(output_grams[l], style_grams[l]) for l in layers
                        ).item()
                        st.markdown(f"- **Style Loss**: `{style_loss:.4f}`")