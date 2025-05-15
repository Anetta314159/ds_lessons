"""
Gatys-style Neural Style Transfer with VGG19 and MLflow Logging.
"""

import os
import io
import torch
from PIL import Image
from torchvision import transforms
import mlflow

from style_transfer.model import StyleTransfer
from style_transfer.utils import image_loader, imsave

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Параметры ===
CONTENT_IMAGE = "data/content.jpg"
STYLE_IMAGE = "data/style.jpg"
SAVE_DIR = "gatys_outputs"
VGG_BACKBONE = "vgg19"
STEPS = 300
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1

os.makedirs(SAVE_DIR, exist_ok=True)

# === Загрузка изображений контента и стиля ===
content_tensor = image_loader(CONTENT_IMAGE, device)
style_tensor = image_loader(STYLE_IMAGE, device)

# === Инициализация style transfer модели ===
transfer = StyleTransfer(content_tensor, style_tensor, device, vgg_name=VGG_BACKBONE)

# === Запуск MLflow сессии ===
mlflow.set_experiment("GatysStyleTransfer")

with mlflow.start_run():
    mlflow.log_param("steps", STEPS)
    mlflow.log_param("style_weight", STYLE_WEIGHT)
    mlflow.log_param("content_weight", CONTENT_WEIGHT)
    mlflow.log_param("vgg_backbone", VGG_BACKBONE)
    mlflow.log_param("content_image", CONTENT_IMAGE)
    mlflow.log_param("style_image", STYLE_IMAGE)

    # === Запуск стиля ===
    output_tensor = transfer.run(
        num_steps=STEPS,
        style_weight=STYLE_WEIGHT,
        content_weight=CONTENT_WEIGHT,
        log_every=50,
        save_dir=SAVE_DIR
    )

    # === Сохранение результата ===
    output_path = os.path.join(SAVE_DIR, "final_output.jpg")
    imsave(output_tensor, output_path)

    # === Логгирование результата в MLflow ===
    mlflow.log_artifact(output_path, artifact_path="generated")

    print(f"✅ Style transfer completed. Result saved to: {output_path}")