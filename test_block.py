import torch
from block import Block # 파일명이 myblock.py이고 같은 폴더에 있다고 가정

# 테스트 입력: 배치 크기 2, 채널 64, 해상도 32x32
x = torch.randn(2, 64, 32, 32)

# 테스트 블록 생성: 다운샘플링 + 출력 채널 수 변경
block = Block(in_ch=64, mid_ch=128, out_ch=96, downsample=True)

# 블록 통과
y = block(x)

# 출력 확인
print("Output shape:", y.shape)