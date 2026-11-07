import numpy as np


def fit_power_law(N, L, floor=0.0):
    """log(L - floor) = log a + b log N을 최소제곱으로 맞춰 (a, b)를 반환한다."""
    N, L = np.asarray(N, dtype=float), np.asarray(L, dtype=float)
    if np.any(L <= floor):
        raise ValueError("모든 손실이 바닥보다 커야 합니다")
    b, log_a = np.polyfit(np.log(N), np.log(L - floor), 1)
    return float(np.exp(log_a)), float(b)


def fit_with_floor(N, L, floors):
    """후보 바닥마다 맞춘 뒤 원래 척도 제곱오차가 가장 작은 (a, b, c)를 반환한다."""
    N, L = np.asarray(N, dtype=float), np.asarray(L, dtype=float)
    best = None
    for c in floors:
        if c >= L.min():
            continue
        a, b = fit_power_law(N, L, c)
        sse = np.sum((L - (a * N**b + c)) ** 2)
        if best is None or sse < best[0]:
            best = (sse, a, b, float(c))
    return best[1], best[2], best[3]
