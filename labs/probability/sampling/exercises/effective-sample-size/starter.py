import numpy as np


def self_normalized_is(logw, fx):
    """로그 가중치 logw (N,)와 함수값 fx (N,)로 자기 정규화 IS 추정값과 ESS를 구한다.

    반환: (estimate float, ess float)
    """
    raise NotImplementedError("overflow 없이 가중치를 정규화하세요")
