import numpy as np


def lj_forces(R):
    """Lennard-Jones 힘 F_i = -dE/dr_i 를 반환한다 (환산 단위 eps = sigma = 1).

    R: (n, 3) 원자 좌표
    반환: (n, 3) 힘
    """
    raise NotImplementedError("쌍마다 힘을 계산해 합하세요")
