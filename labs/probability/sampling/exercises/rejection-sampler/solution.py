import numpy as np


def rejection_sample(density, M, n, rng):
    """[0, 1] 위의 밀도 density에서 균등 제안분포와 envelope 상수 M으로 기각 표집한다.

    반환: (samples (n,), acceptance_rate float)
    """
    kept, proposed, n_kept = [], 0, 0
    while n_kept < n:
        batch = max(2 * (n - n_kept), 1000)
        x = rng.uniform(size=batch)                  # (batch,) 후보
        u = rng.uniform(size=batch)                  # (batch,)
        acc = x[u * M < density(x)]
        kept.append(acc)
        proposed += batch
        n_kept += acc.size
    samples = np.concatenate(kept)[:n]               # (n,)
    return samples, n_kept / proposed
