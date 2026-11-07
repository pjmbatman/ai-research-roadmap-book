import numpy as np


def rejection_sample(density, M, n, rng):
    """[0, 1] 위의 밀도 density에서 균등 제안분포와 envelope 상수 M으로 기각 표집한다.

    반환: (samples (n,), acceptance_rate float)
    """
    raise NotImplementedError("후보를 뽑고 u * M < density(x)인 것만 받아들이세요")
