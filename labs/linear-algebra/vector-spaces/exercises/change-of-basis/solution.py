import numpy as np


def change_basis(A, S):
    """옛 basis에서의 행렬 A를 새 basis에서의 행렬로 바꾼다.

    A: (n, n) 옛 basis에서 T의 행렬
    S: (n, n) 열 = 새 basis 벡터를 옛 basis 좌표로 적은 것 (가역)
    반환: (n, n) 새 basis에서 T의 행렬
    """
    A, S = np.asarray(A, dtype=float), np.asarray(S, dtype=float)
    return np.linalg.solve(S, A @ S)   # S^{-1} A S, (n, n)
