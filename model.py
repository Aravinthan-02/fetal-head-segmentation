import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # Output layer
        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        # e1 shape:
        # [batch, 32, 256, 256]

        p1 = self.pool1(e1)
        # [batch, 32, 128, 128]

        e2 = self.enc2(p1)
        # [batch, 64, 128, 128]

        p2 = self.pool2(e2)
        # [batch, 64, 64, 64]

        # Bottleneck
        b = self.bottleneck(p2)
        # [batch, 128, 64, 64]

        # Decoder level 2
        d2 = self.up2(b)
        # [batch, 64, 128, 128]

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        # [batch, 128, 128, 128]

        d2 = self.dec2(d2)
        # [batch, 64, 128, 128]

        # Decoder level 1
        d1 = self.up1(d2)
        # [batch, 32, 256, 256]

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        # [batch, 64, 256, 256]

        d1 = self.dec1(d1)
        # [batch, 32, 256, 256]

        logits = self.final(d1)
        # [batch, 1, 256, 256]

        return logits