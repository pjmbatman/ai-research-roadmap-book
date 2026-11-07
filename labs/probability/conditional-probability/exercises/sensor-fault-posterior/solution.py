import numpy as np


def posterior_fault(prior, sens, fpr, alarms):
    """센서들이 고장 여부에 대해 조건부 독립일 때 P(고장 | alarms).

    prior: 고장의 사전확률. sens, fpr, alarms: shape (k,). alarms[j]는 0 또는 1.
    """
    sens, fpr = np.asarray(sens, dtype=float), np.asarray(fpr, dtype=float)
    a = np.asarray(alarms).astype(bool)                               # (k,)
    lik1 = np.where(a, sens, 1 - sens)                                # P(a_j | 고장)
    lik0 = np.where(a, fpr, 1 - fpr)                                  # P(a_j | 정상)
    log_odds = np.log(prior) - np.log1p(-prior) + np.sum(np.log(lik1) - np.log(lik0))
    if log_odds >= 0:                                                 # overflow 없는 sigmoid
        return float(1 / (1 + np.exp(-log_odds)))
    e = np.exp(log_odds)
    return float(e / (1 + e))
