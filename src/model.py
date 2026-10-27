# SPDX-License-Identifier: GPL-3.0-only
"""Residual U-Net denoiser.

Attribution
-----------
The building blocks ``DoubleConv``, ``Down`` and ``Up`` and the layer layout of
``UNet`` (``inc``, ``down1``-``down4``, ``up1``-``up4``, ``outc``, the
``factor = 2 if bilinear else 1`` channel rule and the skip-alignment padding)
are adapted from Pytorch-UNet by milesial and contributors,
https://github.com/milesial/Pytorch-UNet (``unet/unet_parts.py`` and
``unet/unet_model.py``), licensed under the GNU General Public License v3.0.

Modifications made for this work:

* BatchNorm2d replaced by GroupNorm (up to 8 groups per layer);
* optional Dropout2d between the two convolutions of each block (p = 0.1);
* channel widths taken from ``Config.features`` (64, 128, 256, 512);
* residual output ``x + res_scale * outc(features)`` with a learnable
  ``res_scale`` initialised to 0.1;
* Kaiming initialisation of all convolutions and a zero-initialised final
  1x1 convolution, so the network is the identity map at initialisation;
* the ``OutConv`` wrapper replaced by a plain 1x1 ``nn.Conv2d`` (``outc``);
* bilinear upsampling selected through ``Config.bilinear`` (default True;
  upstream defaults to transposed convolutions);
* ``use_checkpointing`` removed.

This file, like the rest of this repository, is distributed under the GNU
General Public License v3.0 only; see the LICENSE file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config

class DoubleConv(nn.Module):
    """(Conv -> GroupNorm -> ReLU) × 2 with optional Dropout2d between halves.

    Inner Sequential is stored as `self.double_conv` (not `self.block`) to
    preserve state_dict key compatibility with checkpoints from the original
    notebook.
    """

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, dropout: float = 0.0):
        super().__init__()
        mid_ch = mid_ch or out_ch

        def gn(c: int) -> nn.GroupNorm:
            g = min(8, c)
            while c % g != 0:
                g -= 1
            return nn.GroupNorm(g, c)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False), gn(mid_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False), gn(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """MaxPool(2) -> DoubleConv. Inner Sequential is `self.maxpool_conv`
    (preserving state_dict compatibility with original checkpoints).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsample (bilinear or transposed-conv) -> pad-to-match -> concat skip -> DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, dropout=dropout)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x_up)
        dy = x_skip.size(2) - x_up.size(2)
        dx = x_skip.size(3) - x_up.size(3)
        x_up = F.pad(x_up, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x_skip, x_up], dim=1))

"""Original UNet definition from the publication notebook.
"""

class UNet(nn.Module):
    """Residual U-Net for medical image denoising.

    forward(x):  out = x + res_scale * conv1x1(decoder(encoder(x)))

    Zero-init on the final 1×1 conv => identity at step 0.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        f = cfg.features
        factor = 2 if cfg.bilinear else 1

        self.inc   = DoubleConv(cfg.in_channels, f[0], dropout=cfg.dropout)
        self.down1 = Down(f[0], f[1], dropout=cfg.dropout)
        self.down2 = Down(f[1], f[2], dropout=cfg.dropout)
        self.down3 = Down(f[2], f[3], dropout=cfg.dropout)
        self.down4 = Down(f[3], f[3] * 2 // factor, dropout=cfg.dropout)

        self.up1 = Up(f[3] * 2, f[3] // factor, cfg.bilinear, dropout=cfg.dropout)
        self.up2 = Up(f[3],     f[2] // factor, cfg.bilinear, dropout=cfg.dropout)
        self.up3 = Up(f[2],     f[1] // factor, cfg.bilinear, dropout=cfg.dropout)
        self.up4 = Up(f[1],     f[0],            cfg.bilinear, dropout=cfg.dropout)

        self.outc      = nn.Conv2d(f[0], cfg.out_channels, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # Final conv -> zero => residual = 0 at init => network = identity
        nn.init.zeros_(self.outc.weight)
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        h  = self.up1(x5, x4)
        h  = self.up2(h,  x3)
        h  = self.up3(h,  x2)
        h  = self.up4(h,  x1)
        return x + self.res_scale * self.outc(h)
