import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import pad_channels

class Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, downsample=False, residual=True, zero_last=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

        if zero_last:
            nn.init.zeros_(self.conv4.weight)
            if self.conv4.bias is not None:
                nn.init.zeros_(self.conv4.bias)

        self.use_residual = residual
        self.use_downsample = downsample
        self.down = nn.AvgPool2d(kernel_size=2) if downsample else nn.Identity()

    def forward(self, x):
        out = F.gelu(x)
        out = self.conv1(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = F.gelu(out)
        out = self.conv3(out)
        out = F.gelu(out)
        out = self.conv4(out)

        if self.use_residual:
            out = out + x

        out = self.down(out)
        return out

