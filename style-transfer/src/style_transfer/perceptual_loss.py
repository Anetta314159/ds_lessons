import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def gram_matrix(feature):
    """
    Вычисляет грамм-матрицу для входного тензора.

    Параметры:
        feature (Tensor): Тензор формы (B, C, H, W)

    Возвращает:
        Tensor: Грамм-матрица формы (B, C, C)
    """
    b, c, h, w = feature.size()
    features = feature.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))  # батчевое умножение
    return G / (c * h * w)


class VGGFeatures(nn.Module):
    """
    Извлекает промежуточные признаки из VGG16 по заданным слоям.
    """

    def __init__(self, layers=('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')):
        """
        Аргументы:
            layers (tuple): Названия слоев, признаки которых будут извлекаться
        """
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # Замораживаем веса VGG

        # Словарь: индекс слоя -> имя слоя
        self.layer_map = {
            '0': 'conv1_1', '1': 'relu1_1', '2': 'conv1_2', '3': 'relu1_2',
            '5': 'conv2_1', '6': 'relu2_1', '7': 'conv2_2', '8': 'relu2_2',
            '10': 'conv3_1', '11': 'relu3_1', '12': 'conv3_2', '13': 'relu3_2',
            '14': 'conv3_3', '15': 'relu3_3',
            '17': 'conv4_1', '18': 'relu4_1', '19': 'conv4_2', '20': 'relu4_2',
            '21': 'conv4_3', '22': 'relu4_3'
        }
        self.layers = set(layers)

    def forward(self, x):
        """
        Извлекает признаки с выбранных слоев VGG.

        Аргументы:
            x (Tensor): Входной тензор изображения

        Возвращает:
            dict: Имя слоя -> тензор признаков
        """
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            layer_name = self.layer_map.get(name)
            if layer_name in self.layers:
                features[layer_name] = x
        return features


class PerceptualLoss(nn.Module):
    """
    Перцептуальный лосс для обучения модели быстрого переноса стиля.

    Состоит из content loss (между выходом и контентом) и
    style loss (между грамм-матрицами выходного и стиль-изображения).
    """

    def __init__(self, style_image, style_layers=None, content_layer='relu3_3', device='cpu'):
        """
        Аргументы:
            style_image (Tensor): Изображение стиля [1, 3, H, W]
            style_layers (list): Слои VGG, используемые для style loss
            content_layer (str): Слой VGG для content loss
            device (str): CPU или CUDA
        """
        super().__init__()
        self.vgg = VGGFeatures().to(device)
        self.style_layers = style_layers or ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_layer = content_layer
        self.device = device

        # Предварительно считаем признаки и грамм-матрицы для изображения стиля
        with torch.no_grad():
            self.target_style = self.vgg(style_image)
            self.style_grams = {
                layer: gram_matrix(self.target_style[layer]) for layer in self.style_layers
            }

    def forward(self, output, target):
        """
        Вычисляет content и style лоссы для батча.

        Аргументы:
            output (Tensor): Выход модели [B, 3, H, W]
            target (Tensor): Контент-изображение (ground truth)

        Возвращает:
            tuple: content_loss, style_loss
        """
        out_feats = self.vgg(output)
        tgt_feats = self.vgg(target)

        # Content loss между выходом и контентом
        content_loss = F.mse_loss(out_feats[self.content_layer], tgt_feats[self.content_layer])

        # Style loss между грамм-матрицами
        style_loss = 0.0
        for layer in self.style_layers:
            G = gram_matrix(out_feats[layer])  # Грамм-матрица выхода [B, C, C]
            A = self.style_grams[layer]  # Грамм-матрица стиля [1, C, C]

            # Повторяем A вдоль батча, если нужно
            if A.shape[0] != G.shape[0]:
                A = A.expand(G.shape[0], -1, -1)

            style_loss += F.mse_loss(G, A)

        return content_loss, style_loss