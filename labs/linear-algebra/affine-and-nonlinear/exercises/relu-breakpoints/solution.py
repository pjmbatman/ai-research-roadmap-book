import numpy as np


def breakpoints(c, w, b, tol=1e-9):
    """f(x) = sum_j c_j * ReLU(w_j * x + b_j)의 기울기가 바뀌는 점을 오름차순으로 반환한다.

    c, w, b: 길이 n인 배열
    반환: 1차원 NumPy 배열
    """
    c, w, b = (np.asarray(v, dtype=float) for v in (c, w, b))
    active = (np.abs(w) > tol) & (np.abs(c) > tol)
    points = -b[active] / w[active]
    jumps = c[active] * np.abs(w[active])      # 각 점에서의 기울기 변화

    order = np.argsort(points)
    points, jumps = points[order], jumps[order]

    result = []
    i = 0
    while i < len(points):
        j = i
        total = 0.0
        while j < len(points) and abs(points[j] - points[i]) <= tol:
            total += jumps[j]
            j += 1
        if abs(total) > tol:
            result.append(points[i])
        i = j
    return np.array(result)
