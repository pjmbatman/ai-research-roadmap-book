import numpy as np


def sample_probs(logits, temperature=1.0, top_p=1.0):
    """온도를 적용한 softmax에 nucleus(top-p) 필터를 걸어, 샘플링에 쓸 확률 벡터 (V,)를 반환한다."""
    z = np.asarray(logits, dtype=float) / temperature
    p = np.exp(z - z.max())                       # overflow 방지
    p /= p.sum()
    if top_p >= 1.0:
        return p
    order = np.argsort(-p, kind="stable")         # 확률이 큰 순서
    cum = np.cumsum(p[order])
    n_keep = int(np.searchsorted(cum, top_p)) + 1  # 누적확률이 처음으로 top_p 이상이 되는 위치까지
    keep = np.zeros_like(p, dtype=bool)
    keep[order[:n_keep]] = True
    q = np.where(keep, p, 0.0)
    return q / q.sum()
