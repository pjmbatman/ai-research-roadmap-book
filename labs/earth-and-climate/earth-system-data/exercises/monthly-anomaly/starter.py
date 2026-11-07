import numpy as np


def monthly_anomaly(values, month, base):
    """기준 기간의 월별 기후값과 전체 기간의 편차.

    values: (T, ...), month: (T,) 0–11, base: (T,) bool
    반환: anom (T, ...), clim (12, ...)
    """
    raise NotImplementedError("기후값과 편차를 계산하세요")
