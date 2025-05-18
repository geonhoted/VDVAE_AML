import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
class VAE(nn.Module):
    def __init__(self, encoder, decoder, image_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = image_size  # 예: 32 (CIFAR)

    def forward(self, x, x_target):
        activations = self.encoder(x)  # 해상도별 feature map 반환
        px_z, stats = self.decoder(activations, get_latents=True)

        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)

        # 각 블록에서 KL divergence를 누적
        for stat in stats:
            rate_per_pixel += stat['kl'].sum(dim=(1, 2, 3))

        ndims = np.prod(x.shape[1:])  # 픽셀 수
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean()

        return {
            'elbo': elbo,
            'distortion': distortion_per_pixel.mean(),
            'rate': rate_per_pixel.mean()
        }

    def forward_get_latents(self, x):
        activations = self.encoder(x)
        _, stats = self.decoder(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)