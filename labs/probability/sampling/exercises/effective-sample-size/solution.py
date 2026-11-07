import numpy as np


def self_normalized_is(logw, fx):
    """로그 가중치 logw (N,)와 함수값 fx (N,)로 자기 정규화 IS 추정값과 ESS를 구한다.

    반환: (estimate float, ess float)
    """
    logw = np.asarray(logw, dtype=float)
    fx = np.asarray(fx, dtype=float)
    w = np.exp(logw - logw.max())        # (N,) 최댓값을 빼면 overflow가 없다
    w /= w.sum()                         # 합 1로 정규화
    estimate = np.sum(w * fx)
    ess = 1.0 / np.sum(w**2)             # (Σw)² / Σw², 정규화했으므로 Σw = 1
    return float(estimate), float(ess)
