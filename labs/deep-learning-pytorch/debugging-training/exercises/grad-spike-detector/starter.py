import numpy as np


def find_spikes(norms, window=20, factor=5.0, min_history=5):
    """gradient norm 로그에서 비유한 값과 spike의 step index를 찾는다.

    norms: (T,) step별 global gradient norm
    반환: 이상 step index의 오름차순 list
    """
    raise NotImplementedError("정상 값의 이력을 유지하며 중앙값 기준으로 판정하세요")
