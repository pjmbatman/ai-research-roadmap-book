import numpy as np


def composite_simpson(f, a, b, n):
    """합성 Simpson 규칙으로 ∫_a^b f(x) dx를 근사한다.

    f: 배열을 받아 같은 shape의 배열을 돌려주는 함수
    a, b: 적분 구간
    n: 작은 구간의 수 (2 이상의 짝수). 아니면 ValueError
    반환: float
    """
    if n < 2 or n % 2 != 0:
        raise ValueError(f"n은 2 이상의 짝수여야 합니다 (현재 {n})")
    x = np.linspace(a, b, n + 1)        # (n+1,)
    w = np.ones(n + 1)                  # (n+1,) 가중치 1, 4, 2, 4, ..., 2, 4, 1
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    h = (b - a) / n
    return float(h / 3.0 * np.dot(w, f(x)))
