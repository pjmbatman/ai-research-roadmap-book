import numpy as np


def pagerank(A, d=0.85, tol=1e-12, max_iter=1000):
    """인용 그래프의 PageRank.

    A: (n, n) 배열. A[j, i] = 1이면 논문 i가 논문 j를 인용한다.
    d: damping factor. 참고문헌이 없는 논문의 열은 균등분포로 채운다.
    반환: (n,) 배열, 합은 1.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    out = A.sum(axis=0)                                   # (n,)
    P = np.where(out > 0, A / np.where(out > 0, out, 1.0), 1.0 / n)
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = (1 - d) / n + d * P @ r
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r
