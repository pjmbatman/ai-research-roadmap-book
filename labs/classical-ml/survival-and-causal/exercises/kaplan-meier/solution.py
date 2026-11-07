import numpy as np


def kaplan_meier(time, event):
    """Kaplan–Meier 추정. time: (N,), event: (N,) bool. 반환: (t_j, S_j), 각각 (J,)"""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    t_j = np.unique(time[event])                                       # (J,) 오름차순
    d_j = np.array([np.sum((time == t) & event) for t in t_j])         # 그 시각의 고장 수
    n_j = np.array([np.sum(time >= t) for t in t_j])                   # 위험 집합(같은 시각의 중도절단 포함)
    return t_j, np.cumprod(1.0 - d_j / n_j)


def km_median(t_j, S_j):
    """S가 처음 0.5 이하가 되는 시각. 없으면 np.inf"""
    below = np.flatnonzero(np.asarray(S_j) <= 0.5)
    return float(t_j[below[0]]) if len(below) else np.inf
