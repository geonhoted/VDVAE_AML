# we manage parameters for model structure here.
encblock_str = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"    # encoder 구조
custom_width_str = ""                                        # 해상도별 채널 수 조정 (16:64,8:64)
decblock_str = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"  # decoder 구조

image_size = 32
image_channels = 3
base_width = 384
bottleneck_multiple = 0.25
zdim = 16
num_mixtures = 10
