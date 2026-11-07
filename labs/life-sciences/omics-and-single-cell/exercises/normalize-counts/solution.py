import numpy as np


def normalize_log1p(counts, target_sum=1e4):
    """세포마다 총합을 target_sum으로 맞춘 뒤 log1p.

    counts: (N, G) 음이 아닌 개수. 총합이 0인 세포의 행은 0으로 둔다.
    반환: (N, G) 실수 배열
    """
    C = np.asarray(counts, dtype=float)                 # (N, G) 복사본
    total = C.sum(axis=1, keepdims=True)                # (N, 1)
    total = np.where(total > 0, total, 1.0)             # 빈 세포: 분모 1 → 결과 0
    return np.log1p(C / total * target_sum)
