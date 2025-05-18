import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from block import Block
from utils import draw_gaussian_diag_samples, gaussian_analytical_kl

class DecBlock(nn.Module):
    def __init__(self, res, width, zdim, bottleneck_multiple, mixin_res=None, n_blocks=1):
        super().__init__()
        self.res = res
        self.width = width
        self.zdim = zdim
        self.mixin = mixin_res

        cond_width = int(width * bottleneck_multiple)

        self.enc = Block(width * 2, cond_width, zdim * 2, residual=False)
        self.prior = Block(width, cond_width, zdim * 2 + width, residual=False, zero_last=True)

        self.z_proj = nn.Conv2d(zdim, width, kernel_size=1)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)

        self.resnet = Block(width, cond_width, width, residual=True)
        self.resnet.conv4.weight.data *= np.sqrt(1 / n_blocks)

    def z_fn(self, z):
        return self.z_proj(z)

    def get_inputs(self, xs, activations):
        acts = activations[self.res]
        x = xs.get(self.res, torch.zeros_like(acts))
        
        # 🔥 interpolate acts if shape mismatch
        if acts.shape[2:] != x.shape[2:]:
            acts = F.interpolate(acts, size=x.shape[2:], mode='nearest')

        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)

        return x, acts

    def sample(self, x, acts):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim], feats[:, self.zdim:self.zdim*2], feats[:, self.zdim*2:]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl

    def sample_uncond(self, x, t=None, lvs=None):
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim], feats[:, self.zdim:self.zdim*2], feats[:, self.zdim*2:]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def forward(self, xs, activations, get_latents=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            mix = xs[self.mixin][:, :x.shape[1], ...]
            scale = self.res // self.mixin
            x = x + F.interpolate(mix, scale_factor=scale)

        z, x, kl = self.sample(x, acts)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.res] = x

        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    def forward_uncond(self, xs, t=None, lvs=None):
        if self.res in xs:
            x = xs[self.res]
        else:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(ref.shape[0], self.width, self.res, self.res, device=ref.device)

        if self.mixin is not None:
            mix = xs[self.mixin][:, :x.shape[1], ...]
            scale = self.res // self.mixin
            x = x + F.interpolate(mix, scale_factor=scale)

        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.res] = x
        return xs

