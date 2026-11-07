import numpy as np


def next_point(x_tr, x_cand, ell, sf=1.0, sn=1e-3):
    """GP 사후분산이 가장 큰 후보를 다음 실행점으로 고른다.

    x_tr: (N,) 이미 실행한 입력, x_cand: (M,) 후보 입력
    ell: 길이척도, sf: 신호 표준편차, sn: 잡음 표준편차
    반환: (x_next, var) — x_next는 스칼라, var는 (M,) 사후분산
    """
    x_tr, x_cand = np.asarray(x_tr, dtype=float), np.asarray(x_cand, dtype=float)
    k = lambda a, b: sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)
    L = np.linalg.cholesky(k(x_tr, x_tr) + sn**2 * np.eye(x_tr.size))  # (N, N)
    v = np.linalg.solve(L, k(x_tr, x_cand))                            # (N, M)
    var = sf**2 - np.sum(v**2, axis=0)                                 # (M,)
    return x_cand[np.argmax(var)], var
