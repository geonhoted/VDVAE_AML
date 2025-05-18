import torch
from decoder import Decoder  # 너의 Decoder 정의된 위치
import random

# 디바이스 설정 (GPU 있으면 사용, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ----------- 하이퍼파라미터 설정 -----------
res_list = [(2, None), (4, 2), (8, 4), (16, 8), (32, 16)]  # (res, mixin_res)
width_map = {32: 64, 16: 64, 8: 64, 4: 64, 2: 64}
zdim = 32
bottleneck_multiple = 2
output_res = 32
n_blocks = len(res_list)
num_mixtures = 10
low_bit = False

# ----------- Decoder 인스턴스 생성 -----------
decoder = Decoder(
    res_list=res_list,
    width_map=width_map,
    zdim=zdim,
    bottleneck_multiple=bottleneck_multiple,
    output_res=output_res,
    n_blocks=n_blocks,
    num_mixtures=num_mixtures,
    low_bit=low_bit
).to(device)
decoder.eval()

# ----------- 가짜 activations 생성 (Encoder output 흉내) -----------
B = 2  # batch size
activations = {
    res: torch.randn(B, width_map[res], res, res).to(device)
    for res, _ in res_list
}

# ----------- 테스트 1: forward() (학습용 path) -----------
with torch.no_grad():
    out, stats = decoder.forward(activations, get_latents=True)
    print("✅ forward output:", out.shape)  # e.g., (2, 32, 32, 3)
    for i, stat in enumerate(stats):
        print(f"  • block {i} KL shape:", stat['kl'].shape)

# ----------- 테스트 2: forward_uncond() (샘플 생성) -----------
with torch.no_grad():
    sampled = decoder.forward_uncond(n=B, t=0.8)
    print("✅ unconditional sample:", sampled.shape)  # e.g., (2, 32, 32, 3)

# ----------- 테스트 3: forward_manual_latents() -----------
with torch.no_grad():
    latents = [torch.randn(B, zdim, res, res).to(device) for res, _ in res_list]
    manual_sample = decoder.forward_manual_latents(B, latents, t=0.8)
    print("✅ manual latent sample:", manual_sample.shape)
