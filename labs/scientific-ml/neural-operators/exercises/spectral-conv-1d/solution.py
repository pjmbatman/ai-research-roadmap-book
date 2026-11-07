import numpy as np


def spectral_conv1d(u, weights):
    """FNO의 스펙트럼 합성곱 층.

    u: (C_in, N) 실수, weights: (C_in, C_out, K) 복소수, K <= N//2 + 1
    반환: (C_out, N) 실수. 낮은 K개 Fourier 모드에서만 채널을 섞고 나머지 모드는 버린다.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[-1]
    C_out, K = weights.shape[1], weights.shape[2]
    u_hat = np.fft.rfft(u, axis=-1) / N                               # (C_in, N//2+1), N과 무관한 계수
    v_hat = np.zeros((C_out, N // 2 + 1), dtype=complex)
    v_hat[:, :K] = np.einsum("ik,iok->ok", u_hat[:, :K], weights)     # (C_out, K)
    return np.fft.irfft(v_hat * N, n=N, axis=-1)                      # (C_out, N)
