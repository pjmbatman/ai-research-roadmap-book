import numpy as np


def invariant_features(X):
    """점구름 X (N, 2) → 회전·평행이동·순서 바꾸기에 불변인 1차원 특징 벡터.

    같은 N이면 같은 길이를 반환한다.
    """
    raise NotImplementedError("불변 특징을 설계하세요")
