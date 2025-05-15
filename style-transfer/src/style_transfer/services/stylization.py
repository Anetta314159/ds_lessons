from PIL import Image
from style_transfer.fast_model import TransformNet
from style_transfer.utils import postprocess_batch
import torch
from torchvision import transforms
import os
import io

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stylize_image(content_file, style_file=None, model_name: str = "vangogh_epoch25.pth"):
    """
    Выполняет быстрый перенос стиля с использованием обученной модели (TransformNet).

    Args:
        content_file: Файл с контентным изображением (загруженный через FastAPI/Streamlit).
        style_file: (не используется в Fast Transfer, добавлен для совместимости с интерфейсом).
        model_name (str): Название файла модели в папке `fast_models`.

    Returns:
        PIL.Image: Стилизованное изображение.
    """
    model_path = os.path.join("fast_models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Загрузка модели
    model = TransformNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Загрузка изображения из файла
    image = Image.open(io.BytesIO(content_file.read())).convert('RGB')
    content_file.seek(0)  # ← сброс указателя файла на начало

    # Преобразование изображения: изменение размера, в тензор, нормализация [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    content_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание (без градиентов)
    with torch.no_grad():
        output_tensor = model(content_tensor)
        output_tensor = postprocess_batch(output_tensor)

    # Преобразование результата обратно в PIL.Image
    output_pil = transforms.ToPILImage()(output_tensor.squeeze(0).cpu().clamp(0, 1))
    return output_pil