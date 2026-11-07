import numpy as np


def cka_gram(K, L):
    """두 Gram 행렬로 CKA를 계산한다.

    K, L: (N, N) 같은 입력 N개에 대한 표본 간 유사도 행렬
    반환: float, HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))
    """
    raise NotImplementedError("Gram 행렬을 중심화하고 HSIC 비율을 계산하세요")
