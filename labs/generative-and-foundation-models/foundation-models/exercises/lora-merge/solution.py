import numpy as np


def lora_forward(X, W0, A, B, alpha):
    """병합하지 않은 LoRA 출력 X W0ᵀ + (alpha/r) (X Aᵀ) Bᵀ.

    X: (N, d_in), W0: (d_out, d_in), A: (r, d_in), B: (d_out, r)
    반환: (N, d_out)
    """
    X, W0, A, B = (np.asarray(m, float) for m in (X, W0, A, B))
    r = A.shape[0]
    return X @ W0.T + (alpha / r) * (X @ A.T) @ B.T      # (N, d_out)


def merge(W0, A, B, alpha):
    """병합한 가중치 W0 + (alpha/r) B A, shape (d_out, d_in). W0는 바꾸지 않는다."""
    W0, A, B = (np.asarray(m, float) for m in (W0, A, B))
    return W0 + (alpha / A.shape[0]) * B @ A             # 새 배열을 만든다


def n_trainable(d_in, d_out, r):
    """LoRA가 학습하는 파라미터 수."""
    return int(r * (d_in + d_out))                       # A: r×d_in, B: d_out×r
