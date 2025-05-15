import torch
from torchvision import transforms, models
from PIL import Image
from style_transfer.utils import image_loader, imsave
from style_transfer.model import StyleTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_img = image_loader("data/content.jpg", device)
style_img = image_loader("data/style.jpg", device)

transfer = StyleTransfer(content_img, style_img, device)
output = transfer.run(num_steps=300)

imsave(output, "outputs/output.jpg")
