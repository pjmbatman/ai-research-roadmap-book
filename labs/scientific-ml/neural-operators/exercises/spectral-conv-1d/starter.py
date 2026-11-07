import numpy as np


def spectral_conv1d(u, weights):
    """FNO의 스펙트럼 합성곱 층.

    u: (C_in, N) 실수, weights: (C_in, C_out, K) 복소수, K <= N//2 + 1
    반환: (C_out, N) 실수. 낮은 K개 Fourier 모드에서만 채널을 섞고 나머지 모드는 버린다.
    """
    raise NotImplementedError("rfft → 낮은 K개 모드에서 채널 혼합 → irfft를 구현하세요")
