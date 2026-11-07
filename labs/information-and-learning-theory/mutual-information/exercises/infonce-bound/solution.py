import numpy as np


def infonce(scores):
    """점수 행렬 scores (K, K), scores[i, j] = f(x_i, y_j)에서 InfoNCE 하한 [nat]을 계산한다."""
    S = np.asarray(scores, dtype=float)
    K = S.shape[0]
    m = S.max(axis=1, keepdims=True)                         # (K, 1)
    lse = m[:, 0] + np.log(np.sum(np.exp(S - m), axis=1))    # log Σ_j e^{S_ij}, (K,)
    return float(np.mean(np.diag(S) - (lse - np.log(K))))
