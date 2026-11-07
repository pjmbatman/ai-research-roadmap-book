import hashlib

import numpy as np


def make_streams(seed, names):
    """실험 seed와 용도 이름으로 정해지는 독립 난수 생성기를 만든다.

    seed: int, names: 문자열 리스트
    반환: {이름: np.random.Generator}. 각 생성기는 (seed, 이름)만으로 정해진다.
    """
    raise NotImplementedError("이름을 정수로 바꿔 SeedSequence([seed, 정수])로 생성기를 만드세요")
