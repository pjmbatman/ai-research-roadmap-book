import numpy as np


def _logdet(features, x):
    F = features(x)                                       # (n, p)
    sign, val = np.linalg.slogdet(F.T @ F)
    return val if sign > 0 else -np.inf                   # 특이 행렬은 -inf


def d_optimal(features, cand, n, rng, sweeps=20):
    """교환 알고리즘으로 log det(F^T F)를 최대화하는 실험 조건 n개를 고른다.

    features: (n,) 조건 -> (n, p) 설계 행렬
    cand: (C,) 후보 조건
    n: 실험 수
    rng: np.random.Generator (초기 설계: rng.choice(cand, size=n))
    sweeps: 최대 반복 횟수
    반환: 정렬한 설계 (n,)
    """
    cand = np.asarray(cand, dtype=float)
    x = rng.choice(cand, size=n)
    for _ in range(sweeps):
        improved = False
        for i in range(n):
            scores = [_logdet(features, np.r_[x[:i], c, x[i + 1:]]) for c in cand]
            j = int(np.argmax(scores))
            if scores[j] > _logdet(features, x) + 1e-9:
                x[i], improved = cand[j], True
        if not improved:
            break
    return np.sort(x)
