import numpy as np


def best_split(x, y):
    """특징 하나 x (N,)와 출력 y (N,)에서 제곱오차 합을 최소화하는 문턱을 찾는다.

    반환: (threshold, sse) — x <= threshold가 왼쪽
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]                          # 정렬, (N,)
    n = len(ys)
    cs, cs2 = np.cumsum(ys), np.cumsum(ys**2)            # 왼쪽 i+1개의 합·제곱합
    tot, tot2 = cs[-1], cs2[-1]
    best_t, best_sse = None, np.inf
    for i in range(n - 1):
        if xs[i] == xs[i + 1]:                           # 같은 값 사이는 나눌 수 없다
            continue
        nl, nr = i + 1, n - i - 1
        sse = (cs2[i] - cs[i] ** 2 / nl) + ((tot2 - cs2[i]) - (tot - cs[i]) ** 2 / nr)
        if sse < best_sse - 1e-12:
            best_t, best_sse = (xs[i] + xs[i + 1]) / 2, sse
    return best_t, best_sse
