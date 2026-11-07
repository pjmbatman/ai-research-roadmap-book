import numpy as np


def solution_structure(A, b, tol=1e-10):
    """A x = b의 해가 없는지("none"), 하나인지("unique"), 무수히 많은지("infinite") 판정한다.

    A: (m, n), b: (m,), tol: 수치적 rank의 허용오차
    반환: "none" | "unique" | "infinite"
    """
    raise NotImplementedError("rank(A), rank([A | b]), 열의 수를 비교하세요")
