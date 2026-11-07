import numpy as np


def feasible_pareto(F, G):
    """제약을 만족하는 후보 중 Pareto 최적인 것의 번호를 첫째 목적 오름차순으로 반환한다.

    F: (N, 2) 최소화할 목적, G: (N, m) 제약값(모두 0 이하이면 실현 가능). 반환: (K,) 정수 배열
    """
    F, G = np.asarray(F, dtype=float), np.asarray(G, dtype=float)
    ok = np.flatnonzero(np.all(G <= 0, axis=1))           # 실현 가능한 후보만
    Fk = F[ok]                                            # (K, 2)
    le = np.all(Fk[None, :, :] <= Fk[:, None, :], axis=2)  # le[i, j]: j가 모든 목적에서 i 이하
    lt = np.any(Fk[None, :, :] < Fk[:, None, :], axis=2)   # lt[i, j]: j가 어느 목적에서 i보다 작음
    dominated = np.any(le & lt, axis=1)
    keep = ok[~dominated]
    return keep[np.argsort(F[keep, 0], kind="stable")].astype(int)
