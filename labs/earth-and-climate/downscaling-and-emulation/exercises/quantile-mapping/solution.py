import numpy as np


def quantile_map(x, model_ref, obs_ref, n_q=1001):
    """경험적 quantile mapping: 모델 값 x를 같은 분위의 관측 값으로 바꾼다.

    x: 임의 shape, model_ref: (n,), obs_ref: (m,)
    반환: x와 같은 shape. model_ref 범위 밖의 값은 관측의 끝값으로 고정된다.
    """
    qs = np.linspace(0, 1, n_q)
    mq = np.quantile(np.asarray(model_ref, dtype=float), qs)   # (n_q,)
    oq = np.quantile(np.asarray(obs_ref, dtype=float), qs)     # (n_q,)
    return np.interp(np.asarray(x, dtype=float), mq, oq)
