import numpy as np


def normalize_log1p(counts, target_sum=1e4):
    """세포마다 총합을 target_sum으로 맞춘 뒤 log1p.

    counts: (N, G) 음이 아닌 개수. 총합이 0인 세포의 행은 0으로 둔다.
    반환: (N, G) 실수 배열
    """
    raise NotImplementedError("크기 정규화와 log1p를 구현하세요")
