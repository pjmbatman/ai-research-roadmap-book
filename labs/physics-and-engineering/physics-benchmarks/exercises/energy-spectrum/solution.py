import numpy as np


def energy_spectrum(u, L):
    """길이 L인 주기 영역의 균일 격자 장 u (N,)의 에너지 스펙트럼을 계산한다(N은 짝수).

    반환: k (N//2 + 1,) 파수 2π m / L, E (N//2 + 1,) 모드별 에너지 — sum(E) = mean(u**2) / 2
    """
    u = np.asarray(u, dtype=float)
    N = len(u)
    uh = np.fft.rfft(u) / N                       # (N//2 + 1,)
    E = np.abs(uh) ** 2                           # 0 < m < N/2: 양·음 파수 몫을 합친 값 = 2 × (1/2)|û|²
    E[0] *= 0.5
    E[-1] *= 0.5                                  # N이 짝수일 때 Nyquist 모드는 짝이 없다
    k = 2 * np.pi * np.arange(N // 2 + 1) / L
    return k, E
