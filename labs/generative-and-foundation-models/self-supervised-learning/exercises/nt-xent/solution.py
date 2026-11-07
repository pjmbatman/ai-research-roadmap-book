import numpy as np


def nt_xent(z1, z2, tau):
    """SimCLR의 NT-Xent 손실(nat).

    z1, z2: (N, D) — z1[i]와 z2[i]가 positive 쌍
    tau: temperature (> 0)
    반환: float
    """
    z = np.vstack([np.asarray(z1, float), np.asarray(z2, float)])   # (2N, D)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)                 # 단위 구 위로
    S = z @ z.T / tau                                                # (2N, 2N)
    np.fill_diagonal(S, -np.inf)                                     # 자기 자신 제외
    N = len(z1)
    pos = np.r_[np.arange(N, 2 * N), np.arange(N)]                   # 짝의 인덱스
    m = S.max(1, keepdims=True)
    lse = np.log(np.exp(S - m).sum(1)) + m[:, 0]                     # 안정적인 log-sum-exp
    return float(np.mean(lse - S[np.arange(2 * N), pos]))
