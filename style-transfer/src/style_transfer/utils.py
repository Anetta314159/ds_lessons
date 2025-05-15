from PIL import Image
from torchvision import transforms
import torch
import io

def image_loader(image_source, device, max_size=512):
    """
    Загружает изображение (из пути, файла или байтов) и преобразует в тензор PyTorch.

    Args:
        image_source (str, file-like, or bytes): Путь к изображению, файл или байтовая строка.
        device (torch.device): Устройство для загрузки (CPU или CUDA).
        max_size (int): Размер, до которого будет изменено изображение (по умолчанию 512x512).

    Returns:
        torch.Tensor: Обработанное изображение размером [1, 3, H, W] в диапазоне [-1, 1].
    """
    if isinstance(image_source, str):
        image = Image.open(image_source).convert('RGB')

    elif hasattr(image_source, "read"):
        try:
            image_source.seek(0)  # Сброс указателя на начало файла
        except Exception:
            pass

        image_bytes = image_source.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty or already read.")

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    elif isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source)).convert('RGB')

    else:
        raise ValueError("Unsupported image source type.")

    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # [0,1] → [-1,1]
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def imsave(tensor, out_file):
    """
    Сохраняет тензор изображения в файл JPEG.

    Args:
        tensor (torch.Tensor): Тензор изображения с размером [1, 3, H, W] в диапазоне [-1, 1].
        out_file (str): Путь для сохранения изображения.
    """
    image = tensor.clone().squeeze(0).clamp(-1, 1)
    image = (image + 1) / 2  # [-1,1] → [0,1]
    image = transforms.ToPILImage()(image.cpu())
    image.save(out_file, format='JPEG')

def postprocess_batch(tensor):
    """
    Постобработка батча изображений из [-1, 1] в [0, 1] и обрезка значений.

    Args:
        tensor (torch.Tensor): Тензор изображений.

    Returns:
        torch.Tensor: Обработанный тензор с диапазоном [0, 1].
    """
    tensor = tensor.cpu().clone()
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return tensor