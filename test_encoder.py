import torch
from encoder import Encoder

if __name__ == "__main__":
    # 1. 테스트 입력 (B=2, H=32, W=32, C=3)
    x = torch.randn(2, 32, 32, 3)

    # 2. 테스트용 Encoder 구성
    encoder = Encoder(
        image_channels=3,
        base_width=64,
        custom_width_str="32:64,16:128,8:256",
        block_str="32x2,16d2,8x1",
        bottleneck_multiple=2.0
    )

    # 3. 인코더 통과
    feats = encoder(x)

    # 4. 해상도별 출력 shape 출력
    print("== Feature Map Outputs ==")
    for res, feat in feats.items():
        print(f"Resolution {res} → shape: {feat.shape}")


