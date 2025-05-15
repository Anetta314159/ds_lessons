"""
Обучение сети Fast Style Transfer с логированием в MLflow.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from style_transfer.fast_model import TransformNet
from style_transfer.perceptual_loss import PerceptualLoss
from style_transfer.utils import postprocess_batch, image_loader

# === Настройки параметров ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_name = 'picasso'
style_image_path = f"data/{style_name}.jpg"
dataset_path = "data/coco"
batch_size = 4
epochs = 2
lr = 1e-3
checkpoint_dir = f"checkpoints-{style_name}"
os.makedirs(checkpoint_dir, exist_ok=True)

# === Преобразования и загрузка датасета ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # нормализация в [-1, 1]
])
train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Загрузка изображения стиля ===
style_img = image_loader(style_image_path, device, max_size=256)

# === Инициализация модели и функции потерь ===
model = TransformNet().to(device)
optimizer = optim.Adam(model.parameters(), lr)
loss_fn = PerceptualLoss(style_img, device=device)

# === Инициализация MLflow ===
mlflow.set_experiment("FastStyleTransfer")

with mlflow.start_run():
    # Логгируем гиперпараметры
    mlflow.log_params({
        "style_image": style_image_path,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs
    })

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch, _ in pbar:
            batch = batch.to(device)

            # Прямой проход
            output = model(batch)
            content_loss, style_loss = loss_fn(output, batch)
            total_loss = content_loss + 1e5 * style_loss

            # Шаг оптимизации
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Обновление прогресс-бара
            pbar.set_postfix({
                "content": content_loss.item(),
                "style": style_loss.item()
            })

        # Логгируем метрики в MLflow
        mlflow.log_metrics({
            "content_loss": content_loss.item(),
            "style_loss": style_loss.item()
        }, step=epoch)

        # Сохраняем изображения и модель
        with torch.no_grad():
            out = postprocess_batch(output)
            img_path = os.path.join(checkpoint_dir, f"preview_epoch{epoch+1}.jpg")
            save_image(out, img_path)
            mlflow.log_artifact(img_path, artifact_path=f"previews-{style_name}")

        model_path = os.path.join(checkpoint_dir, f"epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path, artifact_path=f"models-{style_name}")

    # Логгируем финальную модель
    input_example = torch.randn(1, 3, 256, 256).to(device)
    mlflow.pytorch.log_model(model, artifact_path="final_model", input_example=input_example)