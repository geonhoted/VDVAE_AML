import torch
import numpy as np

from vae import VAE
from encoder import Encoder
from decoder import Decoder

# 하이퍼파라미터 설정
image_size = 32
image_channels = 3
base_width = 64
block_str = "32x1,16d2,8d2"                # 인코더 구조
custom_width_str = "16:64,8:64"            # 해상도별 채널 수 조정
bottleneck_multiple = 0.5
zdim = 32
num_mixtures = 10
n_blocks = 1
res_list = [(8, None), (16, 8), (32, 16)]   # 디코더 구조 (res, mixin)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 구성
encoder = Encoder(
    image_channels=image_channels,
    base_width=base_width,
    custom_width_str=custom_width_str,
    block_str=block_str,
    bottleneck_multiple=bottleneck_multiple
)
decoder = Decoder(
    res_list=res_list,
    width_map=encoder.widths,
    zdim=zdim,
    bottleneck_multiple=bottleneck_multiple,
    output_res=image_size,
    n_blocks=n_blocks,
    num_mixtures=num_mixtures,
    low_bit=False
)
vae = VAE(encoder, decoder, image_size=image_size).to(device)

# 더미 데이터 생성 (ex. CIFAR-10)
batch_size = 2
dummy_x = torch.rand(batch_size, image_size, image_size, image_channels).to(device) * 2 - 1  # [-1, 1] 범위
dummy_target = dummy_x.clone()  # reconstruction target은 입력 그대로

# 테스트 실행
vae.eval()
with torch.no_grad():
    result = vae(dummy_x, dummy_target)
    print(f"✅ Using device: {device}")
    print("📌 ELBO:", result['elbo'].item())
    print("📌 Distortion (NLL):", result['distortion'].item())
    print("📌 KL Rate:", result['rate'].item())