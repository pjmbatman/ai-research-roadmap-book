import numpy as np


def lora_forward(X, W0, A, B, alpha):
    """병합하지 않은 LoRA 출력 X W0ᵀ + (alpha/r) (X Aᵀ) Bᵀ.

    X: (N, d_in), W0: (d_out, d_in), A: (r, d_in), B: (d_out, r)
    반환: (N, d_out)
    """
    raise NotImplementedError("고정 경로와 저랭크 경로를 더하세요")


def merge(W0, A, B, alpha):
    """병합한 가중치 W0 + (alpha/r) B A, shape (d_out, d_in). W0는 바꾸지 않는다."""
    raise NotImplementedError("병합한 가중치를 반환하세요")


def n_trainable(d_in, d_out, r):
    """LoRA가 학습하는 파라미터 수."""
    raise NotImplementedError("A와 B의 원소 수를 세세요")
