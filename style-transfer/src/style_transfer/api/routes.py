from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from style_transfer.services.stylization import stylize_image
from style_transfer.model import StyleTransfer
from style_transfer.utils import image_loader, postprocess_batch

import io
import torch
from torchvision.transforms import ToPILImage

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@router.post("/stylize")
async def stylize(
    content: UploadFile = File(...),
    style: UploadFile = File(None),  # Зарезервировано на будущее, не используется в fast-transfer
    model_name: str = Query(..., description="Fast model filename from fast_models/")
):
    """
    Fast style transfer с использованием заранее обученной модели.

    Args:
        content (UploadFile): Контентное изображение.
        style (UploadFile): Не используется (оставлено для совместимости).
        model_name (str): Имя модели из каталога fast_models.

    Returns:
        StreamingResponse: Стилизованное изображение в формате JPEG.
    """
    result_img = stylize_image(content.file, style.file if style else None, model_name)

    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@router.post("/train-gatys")
async def train_gatys_style(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0
):
    """
    Стиль Gatys: перенос стиля с оптимизацией изображения по лоссам.

    Args:
        content (UploadFile): Контентное изображение.
        style (UploadFile): Стилевое изображение.
        steps (int): Количество шагов оптимизации.
        style_weight (float): Вес стилевого лосса.
        content_weight (float): Вес контентного лосса.

    Returns:
        StreamingResponse: Финальное изображение (PNG).
    """
    # Загрузка изображений
    content_tensor = image_loader(content.file, device)
    style_tensor = image_loader(style.file, device)

    # Оптимизация
    transfer = StyleTransfer(content_tensor, style_tensor, device)
    output_tensor = transfer.run(
        num_steps=steps,
        style_weight=style_weight,
        content_weight=content_weight
    )

    # Преобразование в изображение
    output_tensor = postprocess_batch(output_tensor).squeeze(0).cpu().clamp(0, 1)
    pil_img = ToPILImage()(output_tensor)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")