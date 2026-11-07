import numpy as np


def coulomb_eigs(Z, R_bohr):
    """Coulomb 행렬의 고윳값을 큰 것부터 정렬해 반환한다.

    Z: (n,) 원자 번호, R_bohr: (n, 3) 좌표(bohr)
    반환: (n,) 고윳값, 내림차순
    """
    raise NotImplementedError("Coulomb 행렬을 만들고 고윳값을 정렬하세요")
