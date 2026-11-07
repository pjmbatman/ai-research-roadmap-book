import numpy as np


def lj_forces(R):
    """Lennard-Jones 힘 F_i = -dE/dr_i 를 반환한다 (환산 단위 eps = sigma = 1).

    R: (n, 3) 원자 좌표
    반환: (n, 3) 힘
    """
    R = np.asarray(R, dtype=float)
    diff = R[:, None] - R[None]                     # (n, n, 3)
    d = np.linalg.norm(diff, axis=-1)               # (n, n)
    np.fill_diagonal(d, np.inf)                     # 자기 자신 제외
    inv6 = (1 / d) ** 6
    coeff = 24 * (2 * inv6 ** 2 - inv6) / d ** 2    # (n, n)
    return np.einsum("ij,ijk->ik", coeff, diff)     # (n, 3)
