import numpy as np


def softmax_ce(Z, y):
    """batch 평균 softmax cross-entropy와 logit에 대한 gradient.

    Z: (N, K) logits, y: (N,) 정수 레이블
    반환: (L, dZ). L은 스칼라, dZ는 (N, K)
    """
    raise NotImplementedError("손실과 dZ = (P - Y) / N을 계산하세요")
