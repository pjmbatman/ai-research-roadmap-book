import numpy as np


def csr_matvec(data, indices, indptr, x):
    """CSR 세 배열로 주어진 희소 행렬 A에 대해 y = A @ x를 계산한다.

    data: (nnz,), indices: (nnz,), indptr: (m+1,), x: (n,)
    반환: y (m,)
    """
    data, indices, indptr, x = map(np.asarray, (data, indices, indptr, x))
    m = len(indptr) - 1
    y = np.zeros(m, dtype=np.result_type(data.dtype, x.dtype, float))   # (m,)
    for i in range(m):
        lo, hi = indptr[i], indptr[i + 1]
        y[i] = data[lo:hi] @ x[indices[lo:hi]]
    return y
