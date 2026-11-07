import hashlib

import numpy as np


def make_streams(seed, names):
    """실험 seed와 용도 이름으로 정해지는 독립 난수 생성기를 만든다.

    seed: int, names: 문자열 리스트
    반환: {이름: np.random.Generator}. 각 생성기는 (seed, 이름)만으로 정해진다.
    """
    streams = {}
    for name in names:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        key = int.from_bytes(digest[:8], "little")          # 이름 → 64비트 정수(프로세스와 무관)
        streams[name] = np.random.default_rng(np.random.SeedSequence([seed, key]))
    return streams
