import torch
import torch.nn as nn
from utils import parse_layer_string, pad_channels, get_width_settings
from block import Block

class Encoder(nn.Module):
    def __init__(self, image_channels, base_width, custom_width_str, block_str, bottleneck_multiple):
        super().__init__()

        self.in_conv = nn.Conv2d(image_channels, base_width, kernel_size=3, padding=1)
        self.widths = get_width_settings(base_width, custom_width_str)
        block_config = parse_layer_string(block_str)

        enc_blocks = []
        for res, down_rate in block_config:
            width = self.widths[res]
            mid_width = int(width * bottleneck_multiple)

            block = Block(
                in_ch=width,
                mid_ch=mid_width,
                out_ch=width,
                downsample=(down_rate is not None),
                residual=True
            )
            enc_blocks.append(block)

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.block_resolutions = [res for res, _ in block_config]

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)

        feats = {}
        feats[x.shape[2]] = x 

        for block, res in zip(self.enc_blocks, self.block_resolutions):
            if x.shape[1] != self.widths[res]:
                x = pad_channels(x, self.widths[res])

            x = block(x)
            feats[res] = x

        return feats
