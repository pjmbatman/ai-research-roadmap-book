import numpy as np


def fit_ar(x, p):
    """절편 없는 AR(p) 계수를 최소제곱으로 추정한다. x: (T,). 반환: phi (p,)"""
    x = np.asarray(x, dtype=float)
    L = np.column_stack([x[p - k: len(x) - k] for k in range(1, p + 1)])   # (T-p, p)
    phi, *_ = np.linalg.lstsq(L, x[p:], rcond=None)
    return phi


def forecast_ar(x, phi, h):
    """x의 마지막 값까지 보고 h 시점 앞까지 재귀적으로 예측한다. 반환: (h,)"""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    hist = list(np.asarray(x, dtype=float)[-p:])                  # 최근 p개, 오래된 것부터
    out = []
    for _ in range(h):
        nxt = float(phi @ np.array(hist[::-1][:p]))                # phi_1 * 가장 최근 값 + ...
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)
