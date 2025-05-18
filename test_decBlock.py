import torch
from decBlock import DecBlock

if __name__ == "__main__":
    # 설정
    B = 2  # batch size
    res = 16
    width = 128
    zdim = 32
    bottleneck_multiple = 2.0
    n_blocks = 12

    # 가짜 입력 feature들
    xs = {
        res: torch.randn(B, width, res, res)  # 디코더 feature
    }
    activations = {
        res: torch.randn(B, width, res, res)  # 인코더 feature
    }

    # DecBlock 생성
    block = DecBlock(res, width, zdim, bottleneck_multiple, mixin_res=None, n_blocks=n_blocks)

    # forward
    xs_out, out_dict = block(xs, activations, get_latents=True)

    # 출력 확인
    print("📦 최종 출력 feature:", xs_out[res].shape)
    print("📌 z 샘플 shape:", out_dict["z"].shape)
    print("📌 KL shape:", out_dict["kl"].shape)
    print("📊 KL 평균:", out_dict["kl"].mean().item())
