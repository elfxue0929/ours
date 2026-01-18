import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewStem(nn.Module):
    def __init__(self, in_ch: int = 3, c_rgb: int = 64, c_noise: int = 32, c_freq: int = 32):
        super().__init__()
        # RGB branch (reuse-like stem)
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(in_ch, c_rgb, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(c_rgb),
            nn.GELU(),
        )

        # High-pass depthwise conv initialization (Laplacian-like)
        self.highpass = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False
        )
        hp_kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        hp_kernel = hp_kernel.repeat(in_ch, 1, 1, 1)
        with torch.no_grad():
            self.highpass.weight.copy_(hp_kernel)

        # Noise branch (stride 4 total)
        self.noise_branch = nn.Sequential(
            nn.Conv2d(in_ch, c_noise, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(c_noise),
            nn.GELU(),
            nn.Conv2d(c_noise, c_noise, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_noise),
            nn.GELU(),
        )

        # Freq branch (large kernel + group conv). Out channels must be divisible by groups=in_ch
        c_freq_internal = c_freq
        if c_freq_internal % in_ch != 0:
            c_freq_internal = int(math.ceil(c_freq_internal / in_ch) * in_ch)
        
        self.freq_branch = nn.Sequential(
            nn.Conv2d(in_ch, c_freq_internal, kernel_size=8, stride=4, padding=2, groups=in_ch),
            nn.BatchNorm2d(c_freq_internal),
            nn.GELU(),
            nn.Conv2d(c_freq_internal, c_freq, kernel_size=1, stride=1, padding=0),  # Project to desired c_freq
            nn.BatchNorm2d(c_freq),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):  # x: (B, 3, H, W)
        # RGB
        F_rgb = self.rgb_branch(x)  # (B, c_rgb, H/4, W/4)
        # Noise
        x_hp = self.highpass(x)
        F_noise = self.noise_branch(x_hp)  # (B, c_noise, H/4, W/4)
        # Freq
        F_freq = self.freq_branch(x)  # (B, c_freq, ~H/4, ~W/4)
        if F_freq.shape[-2:] != F_rgb.shape[-2:]:
            F_freq = F.interpolate(F_freq, size=F_rgb.shape[-2:], mode="bilinear", align_corners=False)
        return F_rgb, F_noise, F_freq
