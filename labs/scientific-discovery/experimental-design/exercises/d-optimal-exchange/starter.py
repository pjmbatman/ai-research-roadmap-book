import numpy as np


def d_optimal(features, cand, n, rng, sweeps=20):
    """교환 알고리즘으로 log det(F^T F)를 최대화하는 실험 조건 n개를 고른다.

    features: (n,) 조건 -> (n, p) 설계 행렬
    cand: (C,) 후보 조건
    n: 실험 수
    rng: np.random.Generator (초기 설계: rng.choice(cand, size=n))
    sweeps: 최대 반복 횟수
    반환: 정렬한 설계 (n,)
    """
    raise NotImplementedError("한 점씩 후보로 바꿔 보며 log det가 커지면 교환하세요")
