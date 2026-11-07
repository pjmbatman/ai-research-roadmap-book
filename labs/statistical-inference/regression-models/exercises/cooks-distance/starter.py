import numpy as np


def leverage(X):
    """hat 행렬 X (XᵀX)⁻¹ Xᵀ의 대각 성분, shape (n,)."""
    raise NotImplementedError("hat 행렬의 대각 성분을 구한다")


def cooks_distance(X, y):
    """각 관측의 Cook's distance, shape (n,)."""
    raise NotImplementedError("D_i = e_i² / (p σ̂²) · h_ii / (1 - h_ii)²")
