import numpy as np


def grad_check(f, grad, x, eps=1e-6):
    """중앙차분과 비교한 성분별 상대오차의 최댓값.

    f: (n,) -> 스칼라, grad: (n,), x: (n,)
    반환: 스칼라 max_i |g_i - g_num_i| / max(|g_i| + |g_num_i|, 1e-12)
    """
    x = np.array(x, dtype=np.float64)              # 복사본, float64
    g = np.asarray(grad, dtype=np.float64)
    g_num = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        g_num[i] = (f(x + e) - f(x - e)) / (2 * eps)
    rel = np.abs(g - g_num) / np.maximum(np.abs(g) + np.abs(g_num), 1e-12)
    return float(rel.max())
