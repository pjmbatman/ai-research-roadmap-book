import numpy as np


def estimate_order(errors):
    """오차 수열 e_0, ..., e_{n-1}에서 수렴 차수 추정값 p_1, ..., p_{n-2}를 계산한다.

    errors: (n,) 양수 배열, n >= 3
    반환: (n-2,) 배열, p_k = log(e_{k+1}/e_k) / log(e_k/e_{k-1})
    """
    log_e = np.log(np.asarray(errors, dtype=float))   # (n,)
    ratios = np.diff(log_e)                           # (n-1,) log(e_{k+1}/e_k)
    return ratios[1:] / ratios[:-1]                   # (n-2,)
