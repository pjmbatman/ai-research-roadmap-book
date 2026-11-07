import numpy as np


def perplexity(logits, targets):
    """위치별 logit과 정답 토큰으로 perplexity를 계산한다.

    logits: (T, V) 위치마다의 logit
    targets: (T,) 정답 토큰 번호
    반환: float, exp(평균 음의 로그우도)
    """
    z = np.asarray(logits, dtype=float)                       # (T, V)
    y = np.asarray(targets, dtype=int)                        # (T,)
    m = z.max(axis=1, keepdims=True)                          # (T, 1)
    logp = z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))   # (T, V)
    nll = -logp[np.arange(len(y)), y].mean()
    return float(np.exp(nll))
