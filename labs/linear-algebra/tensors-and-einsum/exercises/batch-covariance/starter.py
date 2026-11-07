import numpy as np


def batch_cov(X):
    """X (B, N, D) → batch 항목마다의 표본 공분산 (B, D, D). N-1로 나눈다."""
    raise NotImplementedError("샘플 축으로 중심화한 뒤 einsum으로 공분산을 계산하세요")
