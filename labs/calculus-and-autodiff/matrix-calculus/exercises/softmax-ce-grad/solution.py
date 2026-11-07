import numpy as np


def softmax_ce(Z, y):
    """batch 평균 softmax cross-entropy와 logit에 대한 gradient.

    Z: (N, K) logits, y: (N,) 정수 레이블
    반환: (L, dZ). L은 스칼라, dZ는 (N, K)
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    N, K = Z.shape
    Zs = Z - Z.max(axis=1, keepdims=True)                        # (N, K) overflow 방지
    logp = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))   # (N, K)
    L = -logp[np.arange(N), y].mean()
    Y = np.eye(K)[y]                                             # (N, K) one-hot
    dZ = (np.exp(logp) - Y) / N                                  # (N, K)
    return L, dZ
