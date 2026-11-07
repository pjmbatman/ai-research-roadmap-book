import numpy as np


def top_hessian_eigenvalue(hvp, n, iters=100, seed=0):
    """Hessian-vector product만으로 절댓값이 가장 큰 고윳값과 고유벡터를 추정한다(power iteration).

    hvp: 길이 n인 벡터 v를 받아 H v를 반환하는 함수
    반환: (lam, v) — 고윳값 추정(부호 포함, Rayleigh 몫 v^T H v)과 단위 고유벡터 (n,)
    """
    raise NotImplementedError("power iteration으로 고윳값과 고유벡터를 추정하세요")
