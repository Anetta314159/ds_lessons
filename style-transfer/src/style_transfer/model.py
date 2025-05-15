import torch
import torch.nn.functional as F
from torchvision import models
import os


def gram_matrix(t):
    """
    Вычисляет грамм-матрицу для входного тензора.

    Параметры:
        t (torch.Tensor): тензор размера [B, C, H, W]

    Возвращает:
        torch.Tensor: нормализованная грамм-матрица размера [C*C]
    """
    b, c, h, w = t.size()
    f = t.view(b * c, h * w)
    return torch.mm(f, f.t()) / (b * c * h * w)


def get_vgg(name='vgg19'):
    """
    Загружает VGG-сеть (по умолчанию VGG-19) с предобученными весами.

    Возвращает:
        torch.nn.Sequential: только сверточная часть сети
    """
    return models.vgg19(weights=models.VGG19_Weights.DEFAULT).features


class StyleTransfer:
    """
    Реализация переноса стиля по методу Gatys et al. (2015).

    Использует VGG-сеть для извлечения признаков и оптимизирует изображение напрямую.

    Атрибуты:
        c_img (torch.Tensor): изображение-контент
        s_img (torch.Tensor): изображение-стиль
        device (torch.device): устройство для вычислений
        model (nn.Sequential): предобученная VGG-сеть
        content_layers (list): слои для вычисления content loss
        style_layers (list): слои для style loss
    """

    def __init__(self, content_img, style_img, device, vgg_name='vgg19'):
        self.device = device
        self.c_img = content_img
        self.s_img = style_img
        self.model = get_vgg(vgg_name).to(device).eval()
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_features(self, x):
        """
        Извлекает признаки с выбранных слоев VGG.

        Параметры:
            x (torch.Tensor): входное изображение [B, C, H, W]

        Возвращает:
            dict[str, torch.Tensor]: карта признаков по слоям
        """
        feats = {}
        i = 0
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = f'conv_{i}'
                if name in self.content_layers + self.style_layers:
                    feats[name] = x
        return feats

    def gram_matrix(self, t):
        """
        Обертка для вычисления грамм-матрицы.

        Параметры:
            t (torch.Tensor): входной тензор

        Возвращает:
            torch.Tensor: грамм-матрица
        """
        return gram_matrix(t)

    def run(self, num_steps=300, style_weight=1e6, content_weight=1,
            log_every=50, save_dir=None):
        """
        Запускает оптимизацию изображения для переноса стиля.

        Параметры:
            num_steps (int): количество итераций оптимизации
            style_weight (float): вес стиля
            content_weight (float): вес контента
            log_every (int): шаги, через которые логируются результаты
            save_dir (str): путь для сохранения промежуточных изображений

        Возвращает:
            torch.Tensor: итоговое стилизованное изображение
        """
        input_img = self.c_img.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([input_img])

        run = [0]  # Обертка над счетчиком итераций

        while run[0] <= num_steps:

            def closure():
                # Ограничим значения тензора от -1 до 1
                input_img.data.clamp_(-1, 1)

                optimizer.zero_grad()
                input_features = self.get_features(input_img)
                content_features = self.get_features(self.c_img)
                style_features = self.get_features(self.s_img)

                # Content loss: разница между признаками контента
                content_loss = F.mse_loss(
                    input_features['conv_4'], content_features['conv_4']
                )

                # Style loss: сумма MSE между грамм-матрицами
                style_loss = 0
                for layer in self.style_layers:
                    G = self.gram_matrix(input_features[layer])
                    A = self.gram_matrix(style_features[layer])
                    style_loss += F.mse_loss(G, A)

                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()

                # Logs
                if run[0] % log_every == 0 or run[0] == num_steps:
                    print(f"[Step {run[0]}] Content: {content_loss.item():.4f} | Style: {style_loss.item():.4f}")
                    if save_dir:
                        from style_transfer.utils import imsave, postprocess_batch
                        out = postprocess_batch(input_img.detach())
                        imsave(out, os.path.join(save_dir, f"step_{run[0]:04d}.jpg"))

                run[0] += 1
                return total_loss

            optimizer.step(closure)

        input_img.data.clamp_(-1, 1)
        return input_img.detach()