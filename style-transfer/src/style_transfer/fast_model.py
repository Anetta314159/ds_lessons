import torch
import torch.nn as nn

class ConvLayer(nn.Sequential):
    """
    Слой: Conv2D + InstanceNorm2D + ReLU.
    Используется как базовый строительный блок для энкодера.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2  # "same" padding
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

class ResidualBlock(nn.Module):
    """
    Резидентный блок с двумя сверточными слоями.
    Используется для сохранения деталей контента при обучении.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleConvLayer(nn.Sequential):
    """
    Upsample - Conv2d - InstanceNorm - ReLU
    Используется вместо TransposedConv для повышения стабильности.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
        padding = kernel_size // 2
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*layers)

class TransformNet(nn.Module):
    """
    Основная сеть быстрого переноса стиля:
    - Энкодер: извлекает признаки изображения.
    - Резидентные блоки: сохраняют структуру.
    - Декодер: восстанавливает изображение из признаков.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Encoder
            ConvLayer(3, 32, 9, 1),      # output: [B, 32, H, W]
            ConvLayer(32, 64, 3, 2),     # output: [B, 64, H/2, W/2]
            ConvLayer(64, 128, 3, 2),    # output: [B, 128, H/4, W/4]

            # Residual blocks
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            # Decoder
            UpsampleConvLayer(128, 64, 3, 1, upsample=2),   # [B, 64, H/2, W/2]
            UpsampleConvLayer(64, 32, 3, 1, upsample=2),    # [B, 32, H, W]
            nn.Conv2d(32, 3, 9, 1, 4),                      # [B, 3, H, W]
            nn.Tanh()  # результат в диапазоне [-1, 1]
        )

    def forward(self, x):
        """
        Прямой проход по сети.
        """
        return self.model(x)