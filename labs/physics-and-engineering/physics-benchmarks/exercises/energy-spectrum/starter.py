import numpy as np


def energy_spectrum(u, L):
    """길이 L인 주기 영역의 균일 격자 장 u (N,)의 에너지 스펙트럼을 계산한다(N은 짝수).

    반환: k (N//2 + 1,) 파수 2π m / L, E (N//2 + 1,) 모드별 에너지 — sum(E) = mean(u**2) / 2
    """
    raise NotImplementedError("rfft 계수로 모드별 에너지를 계산하세요")
