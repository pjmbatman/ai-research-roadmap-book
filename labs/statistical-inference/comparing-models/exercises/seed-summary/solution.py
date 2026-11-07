import numpy as np


def summarize(scores):
    """점수 배열의 {"n", "mean", "sd", "se"}. sd는 n-1로 나눈 표본 표준편차, se = sd / sqrt(n)."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    sd = float(s.std(ddof=1))
    return {"n": n, "mean": float(s.mean()), "sd": sd, "se": sd / np.sqrt(n)}


def paired_summary(a, b, t_crit):
    """같은 seed의 점수 쌍에서 d = b - a를 요약한 {"mean_diff", "se", "lo", "hi", "wins_b"}."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean_diff": m, "se": se, "lo": m - t_crit * se, "hi": m + t_crit * se, "wins_b": int(np.sum(b < a))}
