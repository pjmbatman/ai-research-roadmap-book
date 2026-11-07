import numpy as np


def cohen_kappa(a, b):
    """두 평가자의 범주 판정 a, b (길이 N)의 Cohen kappa를 반환한다.

    kappa = (p_o - p_e) / (1 - p_e)
    p_o: 관측 일치율, p_e: 두 평가자의 범주 비율로 계산한 우연 일치율.
    두 평가자가 모두 같은 한 범주만 썼으면 float("nan"), 길이가 다르면 ValueError.
    """
    a, b = np.asarray(a), np.asarray(b)          # (N,), (N,)
    if a.shape != b.shape:
        raise ValueError(f"판정 수가 다릅니다: {a.shape} vs {b.shape}")
    p_o = np.mean(a == b)
    cats = np.union1d(a, b)                      # (K,) 두 평가자가 쓴 범주
    p_e = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    if np.isclose(p_e, 1.0):
        return float("nan")
    return float((p_o - p_e) / (1 - p_e))
