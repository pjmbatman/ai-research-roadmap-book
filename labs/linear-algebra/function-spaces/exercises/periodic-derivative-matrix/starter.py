import numpy as np


def central_diff_matrix(N, L):
    """주기 격자 x_i = i L / N (i = 0..N-1) 위 중심 차분 미분 행렬 (N, N).

    (D u)_i = (u_{i+1} - u_{i-1}) / (2h),  h = L / N,  인덱스는 N을 법으로 돈다.
    """
    raise NotImplementedError("중심 차분 행렬을 만드세요")
