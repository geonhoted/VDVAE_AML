import torch
import torch.nn as nn
from decBlock import DecBlock
from utils import DmolNet 
import itertools

class Decoder(nn.Module):
    def __init__(self, width_map, zdim, bottleneck_multiple,
                 output_res, block_str, num_mixtures, low_bit=False):
        super().__init__()
        self.output_res = output_res
        self.width_map = width_map
        blocks = parse_layer_string(block_str)
        n_blocks = len(blocks)

        self.blocks = nn.ModuleList([
            DecBlock(
                res=res,
                mixin=mixin,
                width=width_map[res],
                zdim=zdim,
                bottleneck_multiple=bottleneck_multiple,
                n_blocks=n_blocks
            )
            for res, mixin in blocks
        ])

        self.bias_xs = nn.ParameterDict({
            str(res): nn.Parameter(torch.zeros(1, width_map[res], res, res))
            for res, _ in blocks
        })

        out_width = width_map[output_res]
        self.gain = nn.Parameter(torch.ones(1, out_width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, out_width, 1, 1))

        self.out_net = DmolNet(width=out_width, num_mixtures=num_mixtures, low_bit=low_bit)

    def final_fn(self, x):
        return x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        B = next(iter(activations.values())).shape[0]
        xs = {
            int(res): bias.repeat(B, 1, 1, 1)
            for res, bias in self.bias_xs.items()
        }

        stats = []
        for block in self.blocks:
            xs, block_stat = block(xs, activations, get_latents=get_latents)
            stats.append(block_stat)

        out = self.final_fn(xs[self.output_res])
        return out, stats 

    def forward_uncond(self, n, t=None):
        xs = {
            int(res): bias.repeat(n, 1, 1, 1)
            for res, bias in self.bias_xs.items()
        }

        for idx, block in enumerate(self.blocks):
            temp = t[idx] if isinstance(t, list) else t
            xs = block.forward_uncond(xs, t=temp)

        out = self.final_fn(xs[self.output_res])
        return self.out_net.sample(out) 

    def forward_manual_latents(self, n, latents, t=None):
        xs = {
            int(res): bias.repeat(n, 1, 1, 1)
            for res, bias in self.bias_xs.items()
        }

        for block, lvs in itertools.zip_longest(self.blocks, latents):
            xs = block.forward_uncond(xs, t=t, lvs=lvs)

        out = self.final_fn(xs[self.output_res])
        return self.out_net.sample(out)
