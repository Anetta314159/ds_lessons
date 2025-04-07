
import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    A simplified UNet architecture for binary image segmentation.

    The network follows an encoder–bottleneck–decoder structure with skip connections.
    It uses two levels of downsampling and upsampling, and outputs a single-channel
    prediction map with values in [0, 1], representing the probability of the foreground class.

    Architecture:
        - Encoder: 2 downsampling blocks (Conv + ReLU + MaxPool)
        - Bottleneck: double conv block
        - Decoder: 2 upsampling blocks with skip connections
        - Output: 1x1 convolution followed by sigmoid activation for binary segmentation

    Input shape:
        (B, 1, H, W) — grayscale image batch

    Output shape:
        (B, 1, H, W) — segmentation probability map
    """
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = CBR(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))
