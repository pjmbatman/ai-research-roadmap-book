import numpy as np


def csr_matvec(data, indices, indptr, x):
    """CSR 세 배열로 주어진 희소 행렬 A에 대해 y = A @ x를 계산한다.

    data: (nnz,), indices: (nnz,), indptr: (m+1,), x: (n,)
    반환: y (m,)
    """
    raise NotImplementedError("행마다 indptr로 구간을 찾아 내적을 계산하세요")
