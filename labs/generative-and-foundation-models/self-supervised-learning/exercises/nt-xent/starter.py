import numpy as np


def nt_xent(z1, z2, tau):
    """SimCLR의 NT-Xent 손실(nat).

    z1, z2: (N, D) — z1[i]와 z2[i]가 positive 쌍
    tau: temperature (> 0)
    반환: float
    """
    raise NotImplementedError("NT-Xent 손실을 계산하세요")
