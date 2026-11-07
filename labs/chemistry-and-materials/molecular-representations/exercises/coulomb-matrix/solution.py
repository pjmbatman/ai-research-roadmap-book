import numpy as np


def coulomb_eigs(Z, R_bohr):
    """Coulomb 행렬의 고윳값을 큰 것부터 정렬해 반환한다.

    Z: (n,) 원자 번호, R_bohr: (n, 3) 좌표(bohr)
    반환: (n,) 고윳값, 내림차순
    """
    Z = np.asarray(Z, dtype=float)
    R = np.asarray(R_bohr, dtype=float)
    D = np.linalg.norm(R[:, None] - R[None], axis=-1)   # (n, n)
    np.fill_diagonal(D, 1.0)
    M = np.outer(Z, Z) / D                               # (n, n)
    np.fill_diagonal(M, 0.5 * Z ** 2.4)
    return np.sort(np.linalg.eigvalsh(M))[::-1]          # (n,)
