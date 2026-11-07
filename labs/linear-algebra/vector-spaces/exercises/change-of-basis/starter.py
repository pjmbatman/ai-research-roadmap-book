import numpy as np


def change_basis(A, S):
    """옛 basis에서의 행렬 A를 새 basis에서의 행렬로 바꾼다.

    A: (n, n) 옛 basis에서 T의 행렬
    S: (n, n) 열 = 새 basis 벡터를 옛 basis 좌표로 적은 것 (가역)
    반환: (n, n) 새 basis에서 T의 행렬
    """
    raise NotImplementedError("기저 변환 공식을 구현하세요")
