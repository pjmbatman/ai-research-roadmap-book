import numpy as np


def filter_and_stress(u, K):
    """고해상도 장 u를 K칸 블록 평균으로 걸러 (ubar, tau)를 반환한다.

    u: (Nf,) 또는 (B, Nf), Nf는 K의 배수
    반환: ubar, tau — 각각 (Nf // K,) 또는 (B, Nf // K), tau = (mean(u^2) - mean(u)^2) / 2
    """
    raise NotImplementedError("블록 평균과 subgrid 응력을 계산하세요")
