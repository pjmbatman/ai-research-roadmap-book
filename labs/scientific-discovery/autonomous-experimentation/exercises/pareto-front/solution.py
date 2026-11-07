import numpy as np


def pareto_mask(F):
    """모든 목적을 최대화할 때 Pareto front에 있는 행을 표시한다.

    F: (N, k) 측정값. 행 = 실험, 열 = 목적(클수록 좋다)
    반환: (N,) 불리언. True = 어떤 실험에도 지배되지 않음
    """
    F = np.asarray(F, dtype=float).reshape(len(F), -1)       # (N, k)
    ge = (F[:, None, :] >= F[None, :, :]).all(-1)             # (N, N): a가 b보다 모든 목적에서 같거나 큼
    gt = (F[:, None, :] > F[None, :, :]).any(-1)              # (N, N): a가 b보다 어떤 목적에서 큼
    dominates = ge & gt                                       # dominates[a, b]: a가 b를 지배
    return ~dominates.any(axis=0)
