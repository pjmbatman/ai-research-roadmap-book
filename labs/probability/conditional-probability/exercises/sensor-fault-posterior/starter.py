import numpy as np


def posterior_fault(prior, sens, fpr, alarms):
    """센서들이 고장 여부에 대해 조건부 독립일 때 P(고장 | alarms).

    prior: 고장의 사전확률. sens, fpr, alarms: shape (k,). alarms[j]는 0 또는 1.
    """
    raise NotImplementedError("로그 오즈에 센서마다 로그 우도비를 더한 뒤 확률로 바꾼다")
