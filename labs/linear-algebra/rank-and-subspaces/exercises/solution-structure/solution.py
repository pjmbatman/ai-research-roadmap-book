import numpy as np


def solution_structure(A, b, tol=1e-10):
    """A x = b의 해가 없는지("none"), 하나인지("unique"), 무수히 많은지("infinite") 판정한다.

    A: (m, n), b: (m,), tol: 수치적 rank의 허용오차
    반환: "none" | "unique" | "infinite"
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    r = np.linalg.matrix_rank(A, tol=tol)
    r_aug = np.linalg.matrix_rank(np.column_stack([A, b]), tol=tol)   # (m, n + 1)
    if r_aug > r:
        return "none"           # b가 column space 밖
    if r == A.shape[1]:
        return "unique"         # null space = {0}
    return "infinite"
