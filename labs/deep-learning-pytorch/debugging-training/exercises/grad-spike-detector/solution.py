import numpy as np


def find_spikes(norms, window=20, factor=5.0, min_history=5):
    """gradient norm 로그에서 비유한 값과 spike의 step index를 찾는다.

    norms: (T,) step별 global gradient norm
    반환: 이상 step index의 오름차순 list
    """
    history, flagged = [], []
    for i, v in enumerate(np.asarray(norms, dtype=float)):
        if not np.isfinite(v):
            flagged.append(i)
            continue
        recent = history[-window:]
        if len(recent) >= min_history and v > factor * np.median(recent):
            flagged.append(i)
            continue                        # 이상 값은 이력에 넣지 않는다
        history.append(v)
    return flagged
