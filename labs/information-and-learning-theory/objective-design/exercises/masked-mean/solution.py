import numpy as np


def masked_token_sum_count(loss, mask):
    """실제 token 손실의 합과 token 수를 (float, int)로 반환한다. loss, mask: (B, T)"""
    mask = np.asarray(mask, dtype=bool)
    clean = np.where(mask, loss, 0.0)                          # padding의 nan·큰 값 제거, (B, T)
    return float(clean.sum()), int(mask.sum())


def masked_token_mean(loss, mask):
    """실제 token에 대한 평균 손실."""
    total, count = masked_token_sum_count(loss, mask)
    return total / count


def masked_sequence_mean(loss, mask):
    """시퀀스별 평균의 평균. 실제 token이 없는 시퀀스는 뺀다."""
    mask = np.asarray(mask, dtype=bool)
    clean = np.where(mask, loss, 0.0)                          # (B, T)
    n = mask.sum(axis=1)                                       # (B,)
    keep = n > 0
    return float(np.mean(clean.sum(axis=1)[keep] / n[keep]))
