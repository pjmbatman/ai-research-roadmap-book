import numpy as np


def batch_cov(X):
    """X (B, N, D) → batch 항목마다의 표본 공분산 (B, D, D). N-1로 나눈다."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)                     # (B, N, D), 항목마다 중심화
    return np.einsum("bnd,bne->bde", Xc, Xc) / (X.shape[1] - 1)  # (B, D, D)
