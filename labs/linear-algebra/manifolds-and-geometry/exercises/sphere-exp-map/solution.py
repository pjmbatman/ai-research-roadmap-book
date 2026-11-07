import numpy as np


def tangent_project(p, x):
    """x (n,)를 단위벡터 p (n,)의 접공간 {v : p·v = 0}으로 직교사영한다."""
    p, x = np.asarray(p, dtype=float), np.asarray(x, dtype=float)
    return x - (p @ x) * p


def sphere_exp(p, v):
    """단위구면 위의 점 p (n,)에서 접벡터 v (n,) 방향으로 측지선을 따라 길이 |v|만큼 간 점 (n,)."""
    p, v = np.asarray(p, dtype=float), np.asarray(v, dtype=float)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return p.copy()
    return np.cos(nv) * p + np.sin(nv) * (v / nv)
